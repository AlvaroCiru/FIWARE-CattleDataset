curl orion:1026/ngsi-ld/v1/subscriptions/ -s -S -H 'Content-Type: application/ld+json' -d @- <<EOF
{
  "description": "Notify changes on all cattle entities",
  "type": "Subscription",
  "entities": [{
    "idPattern": "urn:ngsi-ld:Cattle:.*",
    "type": "Animal"
  }],
  "watchedAttributes": [
    "name",
    "body_temperature",
    "milk_production",
    "respiratory_rate",
    "walking_capacity",
    "sleeping_duration",
    "body_condition_score",
    "heart_rate",
    "eating_duration",
    "lying_down_duration",
    "ruminating",
    "rumen_fill",
    "faecal_consistency",
    "timestamp",
    "activity_ratio",
    "eating_efficiency",
    "vital_sign_index"
  ],
  "notification": {
    "endpoint": {
      "uri": "http://train:4000/notify-cattle",
      "accept": "application/json"
    }
  },
  "@context": [
    "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
    "https://raw.githubusercontent.com/smart-data-models/dataModel.Agrifood/master/context.jsonld"
  ]
}
EOF


curl orion:1026/ngsi-ld/v1/subscriptions/ -s -S -H 'Content-Type: application/ld+json' -d @- <<EOF
{
    "description": "Subscripción a la predicción de salud de Cattle",
    "type": "Subscription",
    "entities": [{
      "id": "urn:ngsi-ld:Cattle:001:Prediction",
      "type": "AnimalPrediction"
    }],
    "watchedAttributes": ["health_status", "confidence", "timestamp"],
    "notification": {
      "endpoint": {
        "uri": "http://train:4000/notify-prediction",
        "accept": "application/json"
      }
    },
    "@context": [
      "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
      "https://raw.githubusercontent.com/smart-data-models/dataModel.Agrifood/master/context.jsonld"
    ]
  }
EOF