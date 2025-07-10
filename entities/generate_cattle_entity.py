import pandas as pd
import numpy as np
import random
from datetime import datetime, timezone
import json

df = pd.read_excel("cattle_dataset.xlsx")  # Asegúrate de tenerlo montado en el contenedor

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

def perturb_row(row):
    noisy = row.copy()
    noisy["heart_rate"] += np.random.normal(0, 5)
    noisy["body_temperature"] += np.random.normal(0, 0.3)
    noisy["eating_duration"] *= np.random.uniform(0.9, 1.1)
    noisy["ruminating"] *= np.random.uniform(0.9, 1.1)
    noisy["faecal_consistency"] = random.choice(df["faecal_consistency"].unique())
    return noisy

def generate_entity():
    row = df.sample(1).iloc[0]
    row = perturb_row(row)
    derived = compute_derived(row)

    entity = {
        "id": "urn:ngsi-ld:Cattle:001",
        "type": "Animal",
        "timestamp": {"type": "Property", "value": datetime.now(timezone.utc).isoformat()},
    }

    for key, val in row.items():
        if key == "faecal_consistency":
            entity[key] = {"type": "Property", "value": str(val)}
        else:
            entity[key] = {"type": "Property", "value": round(float(val), 2)}

    for key, val in derived.items():
        entity[key] = {"type": "Property", "value": round(val, 2)}

    return entity

if __name__ == "__main__":
    entity = generate_entity()
    print(json.dumps(entity, indent=2))
