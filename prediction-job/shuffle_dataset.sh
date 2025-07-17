#!/bin/bash

set -e

INPUT_FILE="cattle_dataset_augmented.csv"
OUTPUT_FILE="cattle_dataset_shuffled.csv"

echo "🔀 Shuffling dataset: $INPUT_FILE -> $OUTPUT_FILE"

# Keep header, shuffle rest
(head -n 1 "$INPUT_FILE" && tail -n +2 "$INPUT_FILE" | shuf) > "$OUTPUT_FILE"

echo "✅ Done. Shuffled dataset saved to: $OUTPUT_FILE"
