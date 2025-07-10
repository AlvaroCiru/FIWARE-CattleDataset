import os
import warnings
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
import mlflow
import time
import emlearn
import threading
import tempfile
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from emlearn.evaluate.trees import model_size_bytes
from flask import Flask, render_template
from flask import request, redirect, url_for, jsonify
import json
from mlflow.tracking import MlflowClient
from datetime import datetime
import requests
from flask_socketio import SocketIO, emit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import shutil
from flask import send_from_directory
from pymongo import MongoClient
from datetime import datetime, timezone


ORION_URL = "http://orion:1026/ngsi-ld/v1"
output_dir = "/mlflow/static_artifacts"
os.makedirs(output_dir, exist_ok=True)

# Cliente hacia mongo-history
history_client = MongoClient("mongodb://root:example@mongo_history:27017/")
history_db = history_client["historicalEntities"]
history_collection = history_db["cowMeasurements"]
experiments_collection = history_db["experiments"]

app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")



def save_experiment(name, start, end, count):
    experiments_collection.insert_one({
        "name": name,
        "start": start,
        "end": end,
        "count": count
    })


def emit_stage(stage):
    print(f"📡 Emitiendo etapa: {stage}")
    socketio.emit("training_stage", {"stage": stage})

# Prepara parámetros para entrenar diferentes algoritmos
def train_with_params(**params):
    def wrapped():
        try:
            train(**params)
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {e}")
    threading.Thread(target=wrapped).start()


# Verifica si un modelo es convertible con Emlearn. max_bytes: tamaño máximo estimado aceptado (~32KB para short int)
def is_emlearn_compatible(model, algorithm, max_bytes=32000):
    try:
        if algorithm in ["RandomForest", "DecisionTree"]:
            cmodel = emlearn.convert(model)
            size = model_size_bytes(cmodel)
            print(f"📏 Emlearn size for {algorithm}: {size} bytes")
            return size <= max_bytes
        else:
            print(f"❌ {algorithm} no es compatible con Emlearn.")
            return False
    except Exception as e:
        print(f"❌ Error al convertir con Emlearn: {e}")
        return False


def eval_metrics(actual, pred):
    return (
        accuracy_score(actual, pred),
        precision_score(actual, pred),
        recall_score(actual, pred),
        f1_score(actual, pred)
    )

def generateCcode(out_dir, model_filename, output_filename):
    code = f'''
    #include "{model_filename}"
    #include <stdio.h>
    int predict(float weekday, float day, float hour, float temperature, float wind_speed) {{
        const float features[5] = {{weekday, day, hour, temperature, wind_speed}};
        const int out = model_predict(features, 5);
        FILE *fptr = fopen("{output_filename}", "w");
        fprintf(fptr, "%d", out);
        fclose(fptr);
        return 0;
    }}
    int main(int argc, const char *argv[]) {{
        return predict(atof(argv[1]), atof(argv[2]), atof(argv[3]), atof(argv[4]), atof(argv[5]));
    }}'''
    code_file = os.path.join(out_dir, 'code.c')
    with open(code_file, 'w') as f:
        f.write(code)
    try:
        emlearn.common.compile_executable(
            code_file, out_dir, name='code', include_dirs=[emlearn.includedir]
        )
    except Exception:
        pass

# Detecta que tipo de modelo se va a entrenar
def get_model(algorithm, **kwargs):
    if algorithm == "RandomForest":
        return RandomForestClassifier(**kwargs)
    elif algorithm == "DecisionTree":
        return DecisionTreeClassifier(**kwargs)
    elif algorithm == "LogisticRegression":
        return LogisticRegression(**kwargs)
    elif algorithm == "LinearRegression":
        return LinearRegression(**kwargs)
    elif algorithm == "NaiveBayes":
        return GaussianNB()
    else:
        raise ValueError(f"Algoritmo no soportado: {algorithm}")

last_run_id = None

# Código de entrenamiento
def train(**params):
    global last_run_id
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Extraer el algoritmo y construir parámetros
    algorithm = params.get("algorithm", "RandomForest")
    model_params = {}
    
    emit_stage("preparing_data")    
    if algorithm == "RandomForest":
        model_params["n_estimators"] = int(params.get("n_estimators", 10))
        model_params["max_depth"] = int(params.get("max_depth", 8))
    elif algorithm == "DecisionTree":
        model_params["max_depth"] = int(params.get("max_depth", 8))
    elif algorithm == "LogisticRegression":
        model_params["max_iter"] = int(params.get("max_iter", 100))
        model_params["C"] = float(params.get("c", 1.0))

    # Datos
    experiment_name = params.get("experiment", None)

    if experiment_name:
        print(f"📡 Usando datos de experimento: {experiment_name}")
        records = list(history_collection.find({"experiment_name": experiment_name}))
        if not records:
            raise ValueError(f"No se encontraron datos para el experimento '{experiment_name}'")
        data = pd.DataFrame(records)
    else:
        print("📄 Usando archivo por defecto: barrier_complete.csv")
        data = pd.read_csv(os.path.join(os.path.dirname(__file__), "barrier_complete.csv"), sep=",")
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=40)

    required_features = ["activity_ratio", "eating_efficiency", "vital_sign_index"]
    target_column = "health_status"

    if target_column not in data.columns:
        raise ValueError("El conjunto de datos no tiene columna 'health_status' para usar como variable objetivo.")

    # Opcional: binariza la clase para clasificación
    data = data[data[target_column].notnull()]
    data[target_column] = data[target_column].apply(lambda x: 1 if x == "unhealthy" else 0)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=40)
    features = required_features


    # MLflow
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    mlflow.set_experiment("FIWARE_MLOPS_TINYML")
    run_name = f"{params.get('experiment', 'default')}_{algorithm}"
    with mlflow.start_run(run_name=run_name) as run:
        last_run_id = run.info.run_id
        with tempfile.TemporaryDirectory() as tmp_dir:

            emit_stage("training_model")
            model = get_model(algorithm, **model_params)
            start = time.time_ns() / 1e6
            model.fit(train_data[features], train_data[target_column])
            end = time.time_ns() / 1e6

            emit_stage("calculating_metrics")
            #preds = model.predict(test_data[features])
            #acc, prec, rec, f1 = eval_metrics(test_data[target_column], preds)

            y_test = test_data[target_column]
            y_pred = model.predict(test_data[features])

            # Métricas básicas
            acc, prec, rec, f1 = eval_metrics(y_test, y_pred)

            # Guardar matriz de confusión
            # Etiquetas con nombres legibles
            labels = ["healthy", "unhealthy"]

            # Matriz normalizada por filas (recall por clase)
            cm = confusion_matrix(y_test, y_pred, normalize="true")

            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(cmap="Greens", ax=ax, colorbar=True)

            # Formato de texto en celdas como porcentajes
            for text in ax.texts:
                try:
                    text.set_text(f"{float(text.get_text()):.2f}")
                except ValueError:
                    pass

            ax.set_title("Normalized Confusion Matrix")
            plt.tight_layout()
            cm_path = os.path.join(tmp_dir, "confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()


            # ROC y AUC (solo si es clasificación binaria y modelo tiene predict_proba)
            if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
                try:
                    y_score = model.predict_proba(test_data[features])[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    auc = roc_auc_score(y_test, y_score)
                    
                    plt.figure()
                    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
                    plt.plot([0, 1], [0, 1], "k--")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("Curva ROC")
                    plt.legend(loc="lower right")
                    roc_path = os.path.join(tmp_dir, "roc_curve.png")
                    plt.savefig(roc_path)
                    plt.close()
                    mlflow.log_artifact(roc_path)
                    mlflow.log_metric("roc_auc", auc)
                except Exception as e:
                    print("⚠️ No se pudo generar la curva ROC:", e)
            # Copia la imagen
            shutil.copy(cm_path, os.path.join(output_dir, f"{run.info.run_id}_confusion_matrix.png"))
            shutil.copy(roc_path, os.path.join(output_dir, f"{run.info.run_id}_roc_curve.png"))
            model_file = os.path.join(tmp_dir, f"{algorithm}_model.joblib")
            dump(model, model_file)

            emit_stage("saving_results")
            mlflow.log_param("algorithm", algorithm)
            for key, value in model_params.items():
                mlflow.log_param(key, value)

            mlflow.sklearn.log_model(model, artifact_path="model")

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("rec", rec)
            mlflow.log_metric("train_time_ms", end - start)
            mlflow.log_artifact(model_file)

            # Intentar convertir y loggear modelo tiny (solo si aplica)
            if is_emlearn_compatible(model, algorithm):
                try:
                    cmodel = emlearn.convert(model)
                    h_path = os.path.join(tmp_dir, f"{algorithm}_model.h")
                    cmodel.save(file=h_path, name="model")
                    result_path = os.path.join(tmp_dir, "result")
                    generateCcode(tmp_dir, h_path, result_path)
                    exe_path = os.path.join(tmp_dir, "code")

                    mlflow.log_artifact(h_path)
                    if os.path.exists(result_path):
                        mlflow.log_artifact(result_path)
                    if os.path.exists(exe_path):
                        mlflow.log_artifact(exe_path)
                        size_kb = os.path.getsize(exe_path) / 1024
                        mlflow.log_metric("model_size_kb", size_kb)
                except Exception as e:
                    print(f"❌ Fallo al compilar modelo Emlearn: {e}")
            else:
                print(f"⚠️ El modelo {algorithm} no es compatible con Emlearn. Se omite conversión.")
            
    emit_stage("completed")




# Endpoint de entrenamiento
@app.route('/train', methods=['POST'])
def train_model():
    data = request.get_json()
    train_with_params(**data)
    return jsonify({'message': 'Entrenamiento lanzado con parámetros personalizados'})

# Endpoint de etado de entrenamiento
@app.route('/train-status', methods=['GET'])
def train_status():
    if last_run_id is None:
        return jsonify({"status": "No training run yet"})

    try:
        client = MlflowClient(tracking_uri="http://mlflow-server:5000")
        run = client.get_run(last_run_id)
        metrics = run.data.metrics
        status = run.info.status
        run_name = run.data.tags.get("mlflow.runName", "Sin nombre")
        algorithm = run.data.params.get("algorithm", "Desconocido")
        run_id = run.info.run_id

        return jsonify({
            "status": status,
            "metrics": metrics,
            "run_name": run_name,
            "algorithm": algorithm,
            "run_id" : run_id,
            "confusion_url": f'http://localhost:4000/static_artifacts/{run_id}_confusion_matrix.png',
            "roc_url": f'http://localhost:4000/static_artifacts/{run_id}_roc_curve.png'

        })

    except Exception as e:
        return jsonify({"status": "Error", "error": str(e)})
    

# Endpoint página de inicio
@app.route('/')
def dashboard():

    try:
        res = requests.get(f"{ORION_URL}/entities/urn:ngsi-ld:Cattle:001:Prediction")
        prediction = res.json() if res.ok else {"id": "Error", "type": f"Status {res.status_code}"}
    except Exception as e:
        prediction = {"id": "Error", "type": str(e)}
    
    try:
        res = requests.get(f"{ORION_URL}/entities/urn:ngsi-ld:Cattle:001")
        cattle = res.json() if res.ok else None
    except Exception as e:
        prediction = {"id": "Error", "type": str(e)}        

    try:
        res = requests.get(f"{ORION_URL}/subscriptions")
        subscriptions = res.json() if res.ok else []
    except Exception as e:
        subscriptions = [{"id": "Error", "type": str(e)}]

    return render_template("dashboard.html", prediction=prediction, subscriptions=subscriptions, cattle = cattle)



# Endpoint mlflow
@app.route('/mlflow-runs')
def mlflow_runs():
    client = MlflowClient(tracking_uri="http://mlflow-server:5000")
    experiment = client.get_experiment_by_name("FIWARE_MLOPS_TINYML")
    if not experiment:
        return jsonify([])

    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=10)
    formatted = []
    for run in runs:
        formatted.append({
            "run_id": run.info.run_id,
            "algorithm": run.data.params.get("algorithm", "N/A"),
            "accuracy": run.data.metrics.get("accuracy_big", run.data.metrics.get("accuracy", "N/A")),
            "start_time": datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return jsonify(formatted)


#Endpoint de entidad de prediccion
@app.route("/notify-prediction", methods=["POST"])
def notify_prediction():
    data = request.get_json()
    socketio.emit("prediction_update", data)
    return "", 204

@app.route("/cattle-history")
def cattle_history():
    try:
        records = list(history_collection.find().sort("timestamp", -1))
        result = []

        for r in records:
            result.append({
                "timestamp": r.get("timestamp", ""),
                "id": str(r.get("id", "")),
                "activity_ratio": r.get("activity_ratio", None),
                "eating_efficiency": r.get("eating_efficiency", None),
                "vital_sign_index": r.get("vital_sign_index", None),
                "health_status": r.get("health_status", "")
            })
        print(result)
        return jsonify(result)  # 👈 ESTO es clave

    except Exception as e:
        print("❌ Error en /cattle-history:", e)

        return jsonify({"error": str(e)}), 500


@app.route("/reset-cattle-history", methods=["POST"])
def reset_cattle_history():
    try:
        deleted = history_collection.delete_many({})
        return {"deleted_count": deleted.deleted_count}, 200
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/notify-cattle", methods=["POST"])
def notify_cattle():
    data = request.get_json()
    socketio.emit("cattle_update", data)

    try:
        print("📥 Datos Cattle recibidos:", json.dumps(data, indent=2))
        cattle_data = data["data"][0]

        record = {
            "timestamp": datetime.now(timezone.utc),
        }

        for key, val in cattle_data.items():
            if isinstance(val, dict) and "value" in val:
                record[key] = val["value"]
            else:
                record[key] = val  # fallback por si es un valor plano

        history_collection.insert_one(record)
        print(f"✅ Insertado historial de vaca con _id: {record.get('_id')}")

    except Exception as e:
        print("❌ Error al guardar historial de vaca:", e)

    return "", 204

@app.route("/simulate-data", methods=["POST"])
def simulate_data():
    from populate import simulate_entries

    try:
        payload = request.get_json()
        name = payload.get("name", "exp")  
        start = payload.get("start")
        end = payload.get("end")
        count = int(payload.get("count", 100))

        simulate_entries(start, end, count, name)
        save_experiment(name, start, end, count)
        return jsonify({"status": "ok", "inserted": int(count)})
    except Exception as e:
        print("❌ Error en /simulate-data:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/get-experiments")
def get_experiments():
    try:
        experiments = list(experiments_collection.find({}, {"_id": 0}))
        return jsonify(experiments)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/delete-experiment", methods=["POST"])
def delete_experiment():
    try:
        data = request.get_json()
        name = data.get("name")

        if not name:
            return jsonify({"status": "error", "message": "Falta nombre del experimento"}), 400

        # Eliminar datos simulados asociados al experimento
        deleted_data = history_collection.delete_many({"experiment_name": name})

        # Eliminar el experimento del registro
        deleted_exp = experiments_collection.delete_one({"name": name})

        return jsonify({
            "status": "ok",
            "deleted_records": deleted_data.deleted_count,
            "deleted_experiment": deleted_exp.deleted_count
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict-cattle", methods=["POST"])
def predict_cattle():
    try:
        from mlflow import sklearn as mlflow_sklearn  # Import aquí por si no se usa en otros sitios

        data = request.get_json()
        run_id = data.get("run_id")
        cow = data.get("cow")  # diccionario con activity_ratio, eating_efficiency, vital_sign_index

        if not run_id or not cow:
            return jsonify({"status": "error", "message": "Falta run_id o datos de la vaca"}), 400

        mlflow.set_tracking_uri("http://mlflow-server:5000")
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)


        # Preparar datos
        features = ["activity_ratio", "eating_efficiency", "vital_sign_index"]
        X = np.array([[cow[f] for f in features]])

        # Predicción
        y_pred = model.predict(X)[0]
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[0].tolist()

        label = "unhealthy" if y_pred == 1 else "healthy"
        timestamp = datetime.now(timezone.utc).isoformat()

        # Notificar a Orion
        payload = {
            "value": {"type": "Property", "value": label},
            "timestamp": {"type": "Property", "value": timestamp}
        }

        if y_proba:
            payload["confidence"] = {"type": "Property", "value": max(y_proba)}

        headers = {"Content-Type": "application/ld+json"}
        orion_entity_url = f"{ORION_URL}/entities/urn:ngsi-ld:DensityDevice:1:Prediction:1/attrs"
        res = requests.patch(orion_entity_url, headers=headers, data=json.dumps(payload))
        print(f"🔄 PATCH Prediction: {res.status_code} - {res.text}")

        # Emitir por socket también
        socketio.emit("prediction_update", {
            "data": [{
                "id": "urn:ngsi-ld:DensityDevice:1:Prediction:1",
                "type": "DeviceMeasurement",
                **payload
            }]
        })

        return jsonify({
            "status": "ok",
            "prediction": label,
            "confidence": max(y_proba) if y_proba else None
        })

    except Exception as e:
        print("❌ Error en /predict-cattle:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/static_artifacts/<filename>")
def static_artifacts(filename):
    return send_from_directory("/mlflow/static_artifacts", filename)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=4000, allow_unsafe_werkzeug=True)