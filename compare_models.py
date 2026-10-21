"""
Compare Logistic Regression and Decision Tree models on a telecom churn dataset.

Usage:
    python compare_models.py
    python compare_models.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv

The CSV should contain a target column named "Churn" (Yes/No), or use --target.
Outputs are saved in the outputs/ folder.
"""
from pathlib import Path
import argparse
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Compare churn classification models.")
    parser.add_argument(
        "--data",
        default=str(PROJECT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        help="Path to the churn dataset CSV."
    )
    parser.add_argument("--target", default="Churn", help="Name of the target column.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def make_one_hot_encoder():
    # Compatibility with both newer and older scikit-learn releases.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main():
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()

    if not data_path.exists():
        print("\nERROR: Dataset CSV was not found.")
        print(f"Expected file: {data_path}")
        print("\nPut your telecom churn CSV in the project's data folder,")
        print("or run with: python compare_models.py --data \"path/to/your.csv\"")
        sys.exit(1)

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        print(f"\nERROR: Target column '{args.target}' was not found.")
        print("Columns in your CSV:")
        print(", ".join(map(str, df.columns)))
        print("\nRun with the correct target name, for example:")
        print('python compare_models.py --target "Exited"')
        sys.exit(1)

    df = df.dropna(subset=[args.target]).copy()
    y_raw = df.pop(args.target)

    # Convert common Yes/No churn labels to 1/0. Other labels are encoded automatically.
    y_text = y_raw.astype(str).str.strip()
    normalized = y_text.str.lower()
    if set(normalized.unique()).issubset({"yes", "no"}):
        y = normalized.map({"no": 0, "yes": 1}).astype(int)
    elif set(normalized.unique()).issubset({"true", "false"}):
        y = normalized.map({"false": 0, "true": 1}).astype(int)
    else:
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y_text), index=y_text.index, name=args.target)
        if y.nunique() != 2:
            print("ERROR: This example expects a binary target with two classes.")
            sys.exit(1)

    # Remove common identifier columns if present; they are not useful predictors.
    identifier_candidates = [
        col for col in df.columns
        if str(col).strip().lower() in {"customerid", "customer_id", "id"}
    ]
    if identifier_candidates:
        df = df.drop(columns=identifier_candidates)

    # Convert numeric-looking columns (e.g. TotalCharges with blank values) to numeric.
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
            non_missing_original = df[col].notna().sum()
            if non_missing_original and converted.notna().sum() >= 0.8 * non_missing_original:
                df[col] = converted

    numeric_features = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in df.columns if c not in numeric_features]

    numeric_logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    numeric_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder())
    ])

    logistic_preprocessor = ColumnTransformer([
        ("numeric", numeric_logistic, numeric_features),
        ("categorical", categorical, categorical_features)
    ])
    tree_preprocessor = ColumnTransformer([
        ("numeric", numeric_tree, numeric_features),
        ("categorical", categorical, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessing", logistic_preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessing", tree_preprocessor),
            ("model", DecisionTreeClassifier(
                random_state=args.random_state, class_weight="balanced",
                max_depth=6, min_samples_leaf=5
            ))
        ])
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1-score": f1_score(y_test, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probabilities)
        })

    results_df = pd.DataFrame(results).set_index("Model")
    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_csv = output_dir / "model_comparison.csv"
    results_df.to_csv(output_csv, float_format="%.4f")

    print("\nMODEL COMPARISON RESULTS")
    print("=" * 76)
    print(results_df.round(4).to_string())
    print(f"\nSaved comparison table to: {output_csv}")

    # Select the best model by F1-score, useful when churn classes are imbalanced.
    best_model = results_df["F1-score"].idxmax()
    best_score = results_df.loc[best_model, "F1-score"]
    conclusion = (
        f"Conclusion: {best_model} performed better based on F1-score "
        f"({best_score:.4f}) on this test split. Review all metrics before choosing "
        "a final model, especially recall and ROC-AUC, because missing customers "
        "who are likely to churn can be costly."
    )
    conclusion_path = output_dir / "conclusion.txt"
    conclusion_path.write_text(conclusion + "\n", encoding="utf-8")
    print("\n" + conclusion)
    print(f"Saved conclusion to: {conclusion_path}")


if __name__ == "__main__":
    main()
