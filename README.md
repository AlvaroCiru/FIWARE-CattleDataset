# FIWARE Machine Learning TinyML and MLOps - Cattle health analysis case

Start base infraestructure
```
docker compose up -d
```
After initialization of the app, you can access the dashboard through your port 4000:

```
http://localhost:4000
```

In the dashboard you will have access to:
  - Simulate data of the Cattle Entity.
  - Train different ML models with this data.
  - Make predictions to determine wether a cow is healthy or unhealthy (using a model previously trained).
  - Watch the different models trained.
  - Access MLFlow for further analysis.
