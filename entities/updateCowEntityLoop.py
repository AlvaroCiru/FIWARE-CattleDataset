#!/usr/bin/env python3
import pandas as pd
import json
import requests
import time
from datetime import datetime, timezone
import numpy as np
import random

df = pd.read_excel('./cattle_dataset.xlsx')

ORION_URL = 'http://orion:1026/ngsi-ld/v1/entities/urn:ngsi-ld:Cattle:001/attrs'
HEADERS = {'Content-Type': 'application/ld+json'}

attribute_mapping = {
    "body_temperature": "body_temperature",
    "milk_production": "milk_production",
    "respiratory_rate": "respiratory_rate",
    "walking_capacity": "walking_capacity",
    "sleeping_duration": "sleeping_duration",
    "body_condition_score": "body_condition_score",
    "heart_rate": "heart_rate",
    "eating_duration": "eating_duration",
    "lying_down_duration": "lying_down_duration",
    "ruminating": "ruminating",
    "rumen_fill": "rumen_fill",
    "faecal_consistency": "faecal_consistency"
}

def compute_derived(row):
    try:
        wc = float(row.get("walking_capacity", 0))
        sd = float(row.get("sleeping_duration", 0))
        ld = float(row.get("lying_down_duration", 0))
        ed = float(row.get("eating_duration", 0))
        ru = float(row.get("ruminating", 0))
        bt = float(row.get("body_temperature", 0))
        rr = float(row.get("respiratory_rate", 0))
        hr = float(row.get("heart_rate", 0))

        activity_ratio = wc / (sd + ld) if (sd + ld) else 0
        eating_efficiency = ed / ru if ru else 0
        vital_sign_index = (bt - 38.5) + ((rr - 30) / 10.0) + ((hr - 60) / 10.0)

        return {
            "activity_ratio": round(activity_ratio, 4),
            "eating_efficiency": round(eating_efficiency, 4),
            "vital_sign_index": round(vital_sign_index, 4)
        }
    except Exception as e:
        print(f"Error during index calculations {e}")
        return {}

def perturb_row(row):
    noisy = row.copy()
    noisy["heart_rate"] += np.random.normal(0, 5)
    noisy["body_temperature"] += np.random.normal(0, 0.3)
    noisy["eating_duration"] *= np.random.uniform(0.9, 1.1)
    noisy["ruminating"] *= np.random.uniform(0.9, 1.1)
    noisy["sleeping_duration"] *= np.random.uniform(0.9, 1.1)
    noisy["lying_down_duration"] *= np.random.uniform(0.9, 1.1)
    noisy["walking_capacity"] *= np.random.uniform(0.9, 1.1)
    noisy["faecal_consistency"] = random.choice(df["faecal_consistency"].unique())
    noisy["milk_production"] += np.random.normal(0, 2)
    noisy["respiratory_rate"] += np.random.normal(0, 3)
    noisy["body_condition_score"] = int(np.clip(round(noisy["body_condition_score"] + np.random.choice([-1, 0, 1])), 1, 5))
    noisy["rumen_fill"] = int(np.clip(round(noisy["rumen_fill"] + np.random.choice([-1, 0, 1])), 1, 5))
    return noisy

def build_payload(row):
    payload = {
        "timestamp": {
            "type": "Property",
            "value": datetime.now(timezone.utc).isoformat()
        }
    }

    # Atributos base
    for csv_col, ngsi_attr in attribute_mapping.items():
        val = row[csv_col]
        if pd.isna(val):
            continue
        if ngsi_attr == "faecal_consistency":
            val = str(val)
        payload[ngsi_attr] = {
            "type": "Property",
            "value": val.item() if hasattr(val, "item") else val
        }

    # Atributos derivados
    derived = compute_derived(row)
    for key, val in derived.items():
        payload[key] = {
            "type": "Property",
            "value": val
        }

    payload["@context"] = [
        "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
        "https://raw.githubusercontent.com/smart-data-models/dataModel.Agrifood/master/context.jsonld"
    ]
    return payload

# Envío periódico
while True:
    row = perturb_row(df.sample(1).iloc[0])
    data = build_payload(row)
    try:
        response = requests.patch(ORION_URL, headers=HEADERS, data=json.dumps(data))
        print(f"[{datetime.now().isoformat()}] PATCH status: {response.status_code}")
        if response.status_code != 204:
            print(f"PATCH failed: {response.text}")
    except Exception as e:
        print(f"Error during PATCH: {e}")
    time.sleep(10)
