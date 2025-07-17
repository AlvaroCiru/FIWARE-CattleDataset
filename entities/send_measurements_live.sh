#!/bin/bash
set -e

echo -e "📡 Starting data simulation from CSV..."

csv_file="cattle_dataset_shuffled.csv"
cow_names=("Mucca" "Charcoal" "Fawn" "Harvest" "Titch" "Silhouette" "Palm" "Brunette" "Annabelle" "Ursula")

start_date="2024-07-01 00:00:00"  # <-- no 'T' or 'Z'
interval_seconds=$((365 * 24 * 60 * 60 / $(wc -l < "$csv_file")))


current_ts=$(date -u -d "$start_date" +%s)
tail -n +2 "$csv_file" | while IFS=',' read -r body_temperature breed_type milk_production respiratory_rate \
  walking_capacity sleeping_duration body_condition_score heart_rate \
  eating_duration lying_down_duration ruminating rumen_fill faecal_consistency \
  health_status activity_ratio eating_efficiency vital_sign_index
do
  random_index=$((RANDOM % 10))
  cow_id=$(printf "urn:ngsi-ld:Cattle:%03d" $((random_index + 1)))
  name="${cow_names[$random_index]}"
  timestamp=$(date -u -d "@$current_ts" +"%Y-%m-%dT%H:%M:%SZ")

  payload=$(cat <<EOF
{
  "id": "$cow_id",
  "type": "Animal",
  "name": { "type": "Property", "value": "$name" },
  "body_temperature": { "type": "Property", "value": $body_temperature },
  "breed_type": { "type": "Property", "value": "$breed_type" },
  "milk_production": { "type": "Property", "value": $milk_production },
  "respiratory_rate": { "type": "Property", "value": $respiratory_rate },
  "walking_capacity": { "type": "Property", "value": $walking_capacity },
  "sleeping_duration": { "type": "Property", "value": $sleeping_duration },
  "body_condition_score": { "type": "Property", "value": $body_condition_score },
  "heart_rate": { "type": "Property", "value": $heart_rate },
  "eating_duration": { "type": "Property", "value": $eating_duration },
  "lying_down_duration": { "type": "Property", "value": $lying_down_duration },
  "ruminating": { "type": "Property", "value": $ruminating },
  "rumen_fill": { "type": "Property", "value": $rumen_fill },
  "faecal_consistency": { "type": "Property", "value": "$faecal_consistency" },
  "health_status": { "type": "Property", "value": "$health_status" },
  "activity_ratio": { "type": "Property", "value": $activity_ratio },
  "eating_efficiency": { "type": "Property", "value": $eating_efficiency },
  "vital_sign_index": { "type": "Property", "value": $vital_sign_index },
  "timestamp": { "type": "Property", "value": "$timestamp" },
  "@context": [
    "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
    "https://raw.githubusercontent.com/smart-data-models/dataModel.Agrifood/master/context.jsonld"
  ]
}
EOF
)

  curl -s -X POST http://orion:1026/ngsi-ld/v1/entities/"$cow_id"/attrs \
    -H "Content-Type: application/ld+json" \
    --data-binary "$payload" > /dev/null

  current_ts=$((current_ts + interval_seconds))
done

echo -e "✅ Finished sending simulated data."
