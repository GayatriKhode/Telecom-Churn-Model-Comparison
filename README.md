# Telecom Customer Churn: Model Comparison

This project compares **Logistic Regression** and **Decision Tree** classifiers on a telecom customer churn dataset using five metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The script creates a comparison table and writes a brief conclusion based on the model with the higher F1-score.

## Project structure

```text
Telecom-Churn-Model-Comparison/
├── compare_models.py
├── requirements.txt
├── README.md
├── data/
│   └── place-your-churn-csv-here.csv
└── outputs/
    ├── model_comparison.csv   (created after running)
    └── conclusion.txt         (created after running)
```

## 1. Add your dataset

Place your telecom churn CSV file inside the `data` folder. The default filename expected by the script is:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

The target column should be named `Churn` and contain values such as `Yes` and `No`. If your file or target column has a different name, use the command options shown below.

**Note:** The dataset is not included in this project ZIP. Use the same dataset from your existing Telecom Customer Churn Prediction project.

## 2. Open the project in VS Code

1. Extract the ZIP file.
2. Open the extracted `Telecom-Churn-Model-Comparison` folder in VS Code.
3. Open **Terminal → New Terminal**.

## 3. Install required libraries

```bash
python -m pip install -r requirements.txt
```

On some Windows installations, use `py` instead of `python` if needed.

## 4. Run the comparison

```bash
python compare_models.py
```

If your CSV has another filename or location:

```bash
python compare_models.py --data "data/your_dataset.csv"
```

If your target column has another name:

```bash
python compare_models.py --data "data/your_dataset.csv" --target "YourTargetColumn"
```

## 5. Check the results

After the script runs, open:

- `outputs/model_comparison.csv` — summary table with the five metrics.
- `outputs/conclusion.txt` — short conclusion based on F1-score.

Scores depend on the dataset and test split. Do not copy example scores from elsewhere; use the scores produced by your own run.

## What the metrics mean

- **Accuracy:** proportion of all predictions that are correct.
- **Precision:** proportion of predicted churners who actually churn.
- **Recall:** proportion of actual churners correctly identified.
- **F1-score:** balance between precision and recall.
- **ROC-AUC:** how well the model separates churners from non-churners across classification thresholds.

## Important notes

- The code uses the same train/test split for both models for a fair comparison.
- Preprocessing is performed inside each model pipeline to reduce data leakage.
- The Decision Tree is limited in depth to help reduce overfitting.
- A higher F1-score is used to select the summary winner, but consider recall and ROC-AUC too before deciding which model is best for your use case.
