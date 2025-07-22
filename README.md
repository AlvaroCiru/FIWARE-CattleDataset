# FIWARE Machine Learning TinyML and MLOps - Cattle health analysis case

Start base infraestructure
```
docker compose up -d
```
After initialization of the app, you can access the dashboard through your port 4000:

```
http://localhost:4000
```
The app will take some seconds to start, and create the historical data for working with the models.

Historical data will be created upon start, including data from the 10 different cattle measured in the past year.

You can train your models selecting data from this time span.

In the dashboard you will have access to:
  - Train different ML models with historical data.
  - Watch in real time the updates on the measurements of the different cows.
  - Watch in real time the predictions made by the models you trained.
  - See the different subscriptions created to communicate with the context broker (Orion-LD).
  - Access MLFlow for further analysis of the models.


## Data Augmentation with Oversampling

To enhance the size of the dataset while keeping class balance, a custom oversampling script has been added. This script generates synthetic data by applying realistic perturbations to existing entries, maintaining biological plausibility.

### Script: `generate_augmented_dataset.py`

This script allows you to generate a specified number of new synthetic samples per class (`healthy` and `unhealthy`) and save the augmented dataset to a file.

### Requirements

Before running the script, make sure to install the required libraries:

```
pip install pandas numpy openpyxl
```

### How to use
```
python generate_augmented_dataset.py \
  --input cattle_dataset.xlsx \
  --output cattle_dataset_augmented.csv \
  --count 350
```
    --input: Path to the original Excel dataset

    --output: Path to save the new augmented dataset (CSV or XLSX)

    --count: Number of synthetic samples to generate per class

This will:

    Load the original dataset

    Generate 2 × count new entries (balanced by class)

    Save the result combining original + synthetic rows

