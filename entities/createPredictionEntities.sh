set -e  # Termina el script si falla cualquier comando

curl orion:1026/ngsi-ld/v1/entities -s -S -H 'Content-Type: application/ld+json' -d @- <<EOF
{
  "id": "urn:ngsi-ld:Cattle:001",
  "type": "Animal",
  "body_temperature": {
    "type": "Property",
    "value": 38.6
  },
  "milk_production": {
    "type": "Property",
    "value": 21.2
  },
  "respiratory_rate": {
    "type": "Property",
    "value": 35
  },
  "walking_capacity": {
    "type": "Property",
    "value": 12000
  },
  "sleeping_duration": {
    "type": "Property",
    "value": 4.2
  },
  "body_condition_score": {
    "type": "Property",
    "value": 3
  },
  "heart_rate": {
    "type": "Property",
    "value": 60
  },
  "eating_duration": {
    "type": "Property",
    "value": 3.5
  },
  "lying_down_duration": {
    "type": "Property",
    "value": 13.4
  },
  "ruminating": {
    "type": "Property",
    "value": 6.1
  },
  "rumen_fill": {
    "type": "Property",
    "value": 4
  },
  "faecal_consistency": {
    "type": "Property",
    "value": "ideal"
  },
  "timestamp": {
    "type": "Property",
    "value": 0
  },
  "activity_ratio": {
    "type": "Property",
    "value": 0
  },
  "eating_efficiency": {
    "type": "Property",
    "value": 0
  },
  "vital_sign_index": {
    "type": "Property",
    "value": 0
  },
  "@context": [
    "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
    "https://raw.githubusercontent.com/smart-data-models/dataModel.Agrifood/master/context.jsonld"
  ]
}
EOF

curl orion:1026/ngsi-ld/v1/entities -s -S -H 'Content-Type: application/ld+json' -d @- <<EOF
  {
    "id": "urn:ngsi-ld:Cattle:001:Prediction",
    "type": "AnimalPrediction",
    "health_status": {
      "type": "Property",
      "value": "unknown"
    },
    "confidence": {
      "type": "Property",
      "value": 0
    },
    "timestamp": {
      "type": "Property",
      "value": "0"
    },
    "@context": [
      "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
      "https://raw.githubusercontent.com/smart-data-models/dataModel.Agrifood/master/context.jsonld"
    ]
  }
EOF
