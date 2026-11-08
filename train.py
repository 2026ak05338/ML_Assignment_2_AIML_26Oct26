from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


# Paths and settings
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATASET_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

TARGET_COLUMN = "y"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def find_local_dataset() -> Path | None:
    """Find bank-full.csv if it has already been downloaded locally."""
    candidates = [
        BASE_DIR / "bank-full.csv",
        BASE_DIR / "data" / "bank-full.csv",
    ]

    for path in candidates:
        if path.exists():
            print(f"Using local dataset: {path}", flush=True)
            return path

    return None


def read_csv_bytes(csv_bytes: bytes) -> pd.DataFrame:
    """Read the UCI Bank Marketing CSV, which uses semicolons."""
    return pd.read_csv(io.BytesIO(csv_bytes), sep=";")


def find_csv_in_zip(archive_bytes: bytes) -> pd.DataFrame:
    """Find bank-full.csv in a ZIP archive or any ZIP nested inside it."""
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = archive.namelist()

        # Check this archive for the CSV.
        for name in names:
            if Path(name).name.lower() == "bank-full.csv":
                print(f"Found dataset in archive: {name}", flush=True)
                return read_csv_bytes(archive.read(name))

        # Search any ZIP files nested inside this archive.
        for name in names:
            if name.lower().endswith(".zip"):
                try:
                    return find_csv_in_zip(archive.read(name))
                except (zipfile.BadZipFile, FileNotFoundError):
                    continue

        raise FileNotFoundError(
            "Could not find bank-full.csv in this ZIP archive. "
            f"Archive entries: {names}"
        )


def get_data() -> pd.DataFrame:
    """Load a local dataset, or download and search the UCI ZIP archive."""
    local_path = find_local_dataset()

    if local_path is not None:
        return pd.read_csv(local_path, sep=";")

    print("Downloading UCI Bank Marketing archive...", flush=True)
    request = Request(
        DATASET_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    try:
        with urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
    except Exception as exc:
        raise RuntimeError(
            "Could not download the UCI dataset. Check internet access, "
            "or place bank-full.csv beside train.py."
        ) from exc

    try:
        return find_csv_in_zip(archive_bytes)
    except (zipfile.BadZipFile, FileNotFoundError) as exc:
        raise RuntimeError(
            "The downloaded archive did not contain bank-full.csv. "
            "You can download bank-full.csv manually and place it beside train.py."
        ) from exc


def make_one_hot_encoder(*, dense: bool = False) -> OneHotEncoder:
    """Create an encoder compatible with newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=not dense,
        )
    except TypeError:
        # sparse_output was called sparse in older scikit-learn versions.
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=not dense,
        )


def make_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    dense: bool = False,
) -> ColumnTransformer:
    """Build imputing, scaling, and one-hot-encoding preprocessing steps."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder(dense=dense)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        # GaussianNB needs dense input; other models can use sparse input.
        sparse_threshold=0.0 if dense else 1.0,
    )


def make_models() -> dict[str, object]:
    """Return the five classifiers used in the assignment."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "svm": CalibratedClassifierCV(
            estimator=LinearSVC(
                random_state=RANDOM_STATE,
                max_iter=5000,
            ),
            cv=3,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        ),
        "naive_bayes": GaussianNB(),
    }


def get_auc(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Calculate ROC-AUC using probabilities or decision scores."""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    else:
        return float("nan")

    return float(roc_auc_score(y_test, scores))


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = get_data().drop_duplicates().reset_index(drop=True)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COLUMN}', "
            f"but found columns: {list(df.columns)}"
        )

    # Convert the target into 0/1 values for consistent binary metrics.
    target = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"no": 0, "yes": 1})
    )

    if target.isna().any():
        raise ValueError(
            f"Unexpected values in target column '{TARGET_COLUMN}': "
            f"{df[TARGET_COLUMN].unique()}"
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = target.astype(int)

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in X.columns if column not in numeric_columns
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Save raw held-out rows and their original target labels.
    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = df.loc[X_test.index, TARGET_COLUMN]
    test_data.to_csv(BASE_DIR / "test_data.csv", index=False)

    results: list[dict[str, float | str]] = []

    for model_name, classifier in make_models().items():
        print(f"Training {model_name}...", flush=True)
        start_time = time.time()

        # GaussianNB requires dense features; the other models use sparse
        # one-hot features to reduce memory usage.
        use_dense = model_name == "naive_bayes"

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_preprocessor(
                        numeric_columns,
                        categorical_columns,
                        dense=use_dense,
                    ),
                ),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = {
            "model": model_name,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": get_auc(pipeline, X_test, y_test),
            "training_seconds": round(time.time() - start_time, 2),
        }
        results.append(metrics)

        model_path = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path)
        print(f"Saved model: {model_path}", flush=True)
        print(
            f"  accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}",
            flush=True,
        )

    metrics_path = BASE_DIR / "metrics.csv"
    pd.DataFrame(results).to_csv(metrics_path, index=False)

    print("\nTraining complete.", flush=True)
    print(f"Metrics saved to: {metrics_path}", flush=True)
    print(f"Test data saved to: {BASE_DIR / 'test_data.csv'}", flush=True)
    print(f"Models saved in: {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
