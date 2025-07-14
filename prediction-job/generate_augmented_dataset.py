import pandas as pd
import numpy as np
import random
import argparse
import os

# === Funciones auxiliares ===
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

def perturb_row(row, df):
    noisy = row.copy()
    noisy["heart_rate"] += np.random.normal(0, 5)
    noisy["body_temperature"] += np.random.normal(0, 0.3)
    noisy["eating_duration"] *= np.random.uniform(0.9, 1.1)
    noisy["ruminating"] *= np.random.uniform(0.9, 1.1)
    noisy["sleeping_duration"] *= np.random.uniform(0.9, 1.1)
    noisy["lying_down_duration"] *= np.random.uniform(0.9, 1.1)
    noisy["walking_capacity"] *= np.random.uniform(0.9, 1.1)
    noisy["faecal_consistency"] = random.choice(df["faecal_consistency"].dropna().unique())
    noisy["milk_production"] += np.random.normal(0, 2)
    noisy["respiratory_rate"] += np.random.normal(0, 3)
    noisy["body_condition_score"] = int(np.clip(round(noisy["body_condition_score"] + random.choice([-1, 0, 1])), 1, 5))
    noisy["rumen_fill"] = int(np.clip(round(noisy["rumen_fill"] + random.choice([-1, 0, 1])), 1, 5))
    return noisy

# === Main ===
def main(input_file, output_file, per_class_count):
    df = pd.read_excel(input_file)

    augmented_rows = []
    for label in ["healthy", "unhealthy"]:
        subset = df[df["health_status"] == label]
        for _ in range(per_class_count):
            row = subset.sample(1).iloc[0]
            perturbed = perturb_row(row, df)
            derived = compute_derived(perturbed)
            full_row = {**perturbed, **derived}
            augmented_rows.append(full_row)

    augmented_df = pd.DataFrame(augmented_rows)
    combined_df = pd.concat([df, augmented_df], ignore_index=True)

    combined_df.to_csv(output_file, index=False)
    print(f"✅ Dataset aumentado guardado en: {output_file}")
    print(f"🔢 Total de registros: {combined_df.shape[0]}")

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar dataset aumentado por oversampling equilibrado.")
    parser.add_argument("--input", type=str, default="cattle_dataset.xlsx", help="Ruta al archivo original .xlsx")
    parser.add_argument("--output", type=str, default="cattle_dataset_augmented.csv", help="Ruta para guardar el nuevo CSV")
    parser.add_argument("--count", type=int, default=300, help="Número de muestras sintéticas por clase")
    args = parser.parse_args()

    main(args.input, args.output, args.count)
