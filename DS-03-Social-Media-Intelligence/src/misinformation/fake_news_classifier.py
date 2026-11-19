
"""LIAR dataset misinformation classification baseline.

This is a research baseline, not an automated fact-checking system.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "external" / "liar"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "misinformation"
MODEL_DIR = ROOT_DIR / "models" / "misinformation"

LABEL_MAP = {
    "pants-fire": "misleading",
    "false": "misleading",
    "barely-true": "misleading",
    "half-true": "not_classified_as_misleading",
    "mostly-true": "not_classified_as_misleading",
    "true": "not_classified_as_misleading",
}

CLASS_NAMES = [
    "misleading",
    "not_classified_as_misleading",
]


def load_dataset(filename):
    """Load LIAR TSV data. Column 1 is the label; column 2 is the statement."""

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        encoding="utf-8",
        quoting=3,
        on_bad_lines="warn",
    )

    if df.shape[1] < 3:
        raise ValueError(f"Unexpected LIAR dataset format: {filename}")

    data = pd.DataFrame({
        "original_label": df.iloc[:, 1].astype(str).str.strip(),
        "text": df.iloc[:, 2].fillna("").astype(str).str.strip(),
    })

    data["label"] = data["original_label"].map(LABEL_MAP)

    data = data.dropna(subset=["label"])
    data = data[data["text"].str.len() > 0]
    data = data.drop_duplicates(subset=["text"])

    return data.reset_index(drop=True)


def build_pipeline():
    """Create TF-IDF + Logistic Regression pipeline."""

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                stop_words="english",
                ngram_range=(1, 2),
                max_features=50000,
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def evaluate_model(model, data, split_name):
    """Calculate evaluation metrics and save confusion matrix."""

    predictions = model.predict(data["text"])
    actual = data["label"]

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predictions,
        labels=CLASS_NAMES,
        average="binary",
        pos_label="misleading",
        zero_division=0,
    )

    report = classification_report(
        actual,
        predictions,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    matrix = confusion_matrix(
        actual,
        predictions,
        labels=CLASS_NAMES,
    )

    metrics = {
        "split": split_name,
        "samples": int(len(data)),
        "accuracy": float(accuracy_score(actual, predictions)),
        "misleading_precision": float(precision),
        "misleading_recall": float(recall),
        "misleading_f1": float(f1),
        "class_balance": {
            str(label): int(count)
            for label, count in actual.value_counts().items()
        },
        "confusion_matrix": matrix.tolist(),
    }

    print(f"\n{'=' * 55}")
    print(f"{split_name.upper()} RESULTS")
    print(f"{'=' * 55}")
    print(json.dumps(metrics, indent=4))
    print("\nClassification report:")
    print(report)

    (OUTPUT_DIR / f"{split_name}_classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(OUTPUT_DIR / f"{split_name}_confusion_matrix.csv")

    return metrics


def train_baseline():
    """Train on train.tsv, validate on valid.tsv, test on test.tsv."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading LIAR dataset...")

    train_data = load_dataset("train.tsv")
    valid_data = load_dataset("valid.tsv")
    test_data = load_dataset("test.tsv")

    print(f"Training samples:   {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")
    print(f"Test samples:       {len(test_data)}")

    print("\nTraining class balance:")
    print(train_data["label"].value_counts())

    model = build_pipeline()

    print("\nTraining TF-IDF + Logistic Regression...")
    model.fit(train_data["text"], train_data["label"])

    print("\nEvaluating validation split...")
    validation_metrics = evaluate_model(
        model, valid_data, "validation"
    )

    print("\nEvaluating test split...")
    test_metrics = evaluate_model(
        model, test_data, "test"
    )

    model_path = MODEL_DIR / "liar_tfidf_logistic_regression.joblib"
    joblib.dump(model, model_path)

    report = {
        "dataset": "LIAR",
        "model": "TF-IDF + Logistic Regression",
        "label_mapping": LABEL_MAP,
        "positive_class": "misleading",
        "validation": validation_metrics,
        "test": test_metrics,
        "limitations": [
            "Binary labels are simplified groupings of LIAR ratings.",
            "The model predicts dataset label patterns, not verified truth.",
            "Predictions require independent fact-checking.",
            "Performance may not generalize to YouTube comments.",
        ],
    }

    report_path = OUTPUT_DIR / "misinformation_metrics.json"
    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nTraining completed successfully.")
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {report_path}")


if __name__ == "__main__":
    train_baseline()
