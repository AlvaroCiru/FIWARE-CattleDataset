import pandas as pd
import random
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
import numpy as np

df = pd.read_excel("cattle_dataset.xlsx")

def compute_derived(row):
    wc = row["walking_capacity"]
    sd = row["sleeping_duration"]
    ld = row["lying_down_duration"]
    ed = row["eating_duration"]
    ru = row["ruminating"]
    bt = row["body_temperature"]
    rr = row["respiratory_rate"]
    hr = row["heart_rate"]

    return {
        "activity_ratio": round(wc / (sd + ld), 4) if (sd + ld) else 0,
        "eating_efficiency": round(ed / ru, 4) if ru else 0,
        "vital_sign_index": round((bt - 38.5) + ((rr - 30) / 10) + ((hr - 60) / 10), 4)
    }

def to_native(val):
    if isinstance(val, (np.integer, np.int64)): return int(val)
    if isinstance(val, (np.floating, np.float64)): return float(val)
    return val

def simulate_entries(start_str, end_str, count, name="exp"):
    client = MongoClient("mongodb://root:example@mongo_history:27017/")
    collection = client["historicalEntities"]["cowMeasurements"]
    
    start = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(end_str).replace(tzinfo=timezone.utc)
    delta = int((end - start).total_seconds())

    for i in range(count):
        row = df.sample(1).iloc[0]
        ts = start + timedelta(seconds=random.randint(0, delta))
        record = {
            "_id": f"{name}_{i}",
            "experiment_name": name,
            "id": "urn:ngsi-ld:Cattle:001",  # <-- Añadir ID fijo
            "timestamp": ts.isoformat(),     # <-- Timestamp como string ISO 8601
        }

        for col in df.columns:
            val = row[col]
            record[col] = str(val) if col == "faecal_consistency" else to_native(val)

        # Añade campos derivados y convierte todos los valores
        derived = compute_derived(row)
        record.update({k: to_native(v) for k, v in derived.items()})

        try:
            collection.insert_one(record)
        except Exception as e:
            print(f"⚠️ Error insertando {record['_id']}: {e}")

    print(f"✅ Insertados {count} registros simulados para experimento '{name}'")

