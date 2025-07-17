#!/bin/bash

set -e

echo -e "⏳ Creating 10 cattle entities in NGSI-LD format"

# Lista de nombres fijos para vacas
cow_names=("Mucca" "Charcoal" "Fawn" "Harvest" "Titch" "Silhouette" "Palm" "Brunette" "Annabelle" "Ursula")

animals="["
index=1

while IFS=',' read -r body_temperature breed_type milk_production respiratory_rate \
  walking_capacity sleeping_duration body_condition_score heart_rate \
  eating_duration lying_down_duration ruminating rumen_fill faecal_consistency \
  health_status activity_ratio eating_efficiency vital_sign_index
do
  # Omitir encabezado
  if [[ "$body_temperature" == "body_temperature" ]]; then
    continue
  fi

  # Validación: sustituir campos vacíos o NaN por 0
  for var in body_temperature milk_production respiratory_rate walking_capacity sleeping_duration \
    body_condition_score heart_rate eating_duration lying_down_duration ruminating rumen_fill \
    activity_ratio eating_efficiency vital_sign_index
  do
    eval "val=\$$var"
    if [[ -z "$val" || "$val" == "NaN" ]]; then
      eval "$var=0"
    fi
  done

  cow_id=$(printf "urn:ngsi-ld:Cattle:%03d" "$index")
  name="${cow_names[$((index-1))]}"

  animals="$animals
  {
    \"id\": \"$cow_id\",
    \"type\": \"Animal\",
    \"name\": { \"type\": \"Property\", \"value\": \"$name\" },
    \"body_temperature\": { \"type\": \"Property\", \"value\": $body_temperature },
    \"breed_type\": { \"type\": \"Property\", \"value\": \"$breed_type\" },
    \"milk_production\": { \"type\": \"Property\", \"value\": $milk_production },
    \"respiratory_rate\": { \"type\": \"Property\", \"value\": $respiratory_rate },
    \"walking_capacity\": { \"type\": \"Property\", \"value\": $walking_capacity },
    \"sleeping_duration\": { \"type\": \"Property\", \"value\": $sleeping_duration },
    \"body_condition_score\": { \"type\": \"Property\", \"value\": $body_condition_score },
    \"heart_rate\": { \"type\": \"Property\", \"value\": $heart_rate },
    \"eating_duration\": { \"type\": \"Property\", \"value\": $eating_duration },
    \"lying_down_duration\": { \"type\": \"Property\", \"value\": $lying_down_duration },
    \"ruminating\": { \"type\": \"Property\", \"value\": $ruminating },
    \"rumen_fill\": { \"type\": \"Property\", \"value\": $rumen_fill },
    \"faecal_consistency\": { \"type\": \"Property\", \"value\": \"$faecal_consistency\" },
    \"health_status\": { \"type\": \"Property\", \"value\": \"$health_status\" },
    \"activity_ratio\": { \"type\": \"Property\", \"value\": $activity_ratio },
    \"eating_efficiency\": { \"type\": \"Property\", \"value\": $eating_efficiency },
    \"vital_sign_index\": { \"type\": \"Property\", \"value\": $vital_sign_index },
    \"@context\": [
      \"https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld\",
      \"https://raw.githubusercontent.com/smart-data-models/dataModel.Agrifood/master/context.jsonld\"
    ]
  },"

  index=$((index+1))
  [[ "$index" -gt 10 ]] && break
done < cattle_dataset_augmented.csv

# Eliminar la última coma y cerrar el array
animals="${animals::-1}
]"

echo -e "📄 Writing JSON to file cattle_entities.json..."
echo "$animals" > cattle_entities.json

echo -e "📡 Sending entities to Orion..."

curl -X POST \
  'http://orion:1026/ngsi-ld/v1/entityOperations/upsert?options=update' \
  -H 'Content-Type: application/ld+json' \
  --data-binary @cattle_entities.json

echo -e "✅ Done: 10 cattle entities created."
