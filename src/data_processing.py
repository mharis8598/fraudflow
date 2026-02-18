"""
Data processing pipeline for credit card fraud detection.
Handles loading, cleaning, splitting, and resampling.
"""

import logging
import os

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_data(filepath: str | None = None) -> pd.DataFrame:
    """Load raw credit card transaction data.

    Args:
        filepath: Path to CSV file. Defaults to data/raw/creditcard.csv.

    Returns:
        Raw DataFrame with all features and target.
    """
    if filepath is None:
        filepath = os.path.join(RAW_DATA_DIR, "creditcard.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "Place creditcard.csv in data/raw/"
        )

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} transactions from {filepath}")
    logger.info(f"Fraud rate: {df['Class'].mean():.4%} ({df['Class'].sum()} fraudulent)")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the dataset.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame with no nulls or duplicates.
    """
    initial_rows = len(df)

    # Drop duplicates
    df = df.drop_duplicates()
    dupes_removed = initial_rows - len(df)
    if dupes_removed > 0:
        logger.info(f"Removed {dupes_removed:,} duplicate rows")

    # Check for nulls
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        logger.warning(f"Found {null_count} null values — dropping rows")
        df = df.dropna()

    # Validate target column
    assert set(df["Class"].unique()) == {0, 1}, "Target must be binary (0, 1)"

    logger.info(f"Clean dataset: {len(df):,} rows, {df.shape[1]} columns")
    return df.reset_index(drop=True)


def preprocess_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """Scale Amount and Time features, return X and y.

    The V1-V28 features are already PCA-transformed.
    Only Amount and Time need scaling.

    Args:
        df: Cleaned DataFrame.
        scaler: Existing scaler for inference. None to create new.
        fit_scaler: Whether to fit the scaler (True for training).

    Returns:
        Tuple of (X features, y target, fitted scaler).
    """
    X = df.drop("Class", axis=1).copy()
    y = df["Class"].copy()

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])
    else:
        X[["Amount", "Time"]] = scaler.transform(X[["Amount", "Time"]])

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y, scaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Proportion for test set.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        f"Train: {len(X_train):,} samples "
        f"(fraud: {y_train.sum():,} = {y_train.mean():.4%})"
    )
    logger.info(
        f"Test:  {len(X_test):,} samples "
        f"(fraud: {y_test.sum():,} = {y_test.mean():.4%})"
    )
    return X_train, X_test, y_train, y_test


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to balance classes in training data only.

    Args:
        X_train: Training features.
        y_train: Training target.
        random_state: Seed for reproducibility.

    Returns:
        Resampled (X_train, y_train) with balanced classes.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    logger.info(
        f"SMOTE applied: {len(X_train):,} → {len(X_resampled):,} samples "
        f"(fraud: {y_resampled.sum():,} = {y_resampled.mean():.2%})"
    )
    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)


def run_pipeline(filepath: str | None = None) -> dict:
    """Execute the full data processing pipeline.

    Args:
        filepath: Optional path to raw CSV.

    Returns:
        Dictionary with all processed data splits and scaler.
    """
    df = load_data(filepath)
    df = clean_data(df)
    X, y, scaler = preprocess_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    return {
        "X_train": X_train_sm,
        "X_test": X_test,
        "y_train": y_train_sm,
        "y_test": y_test,
        "X_train_original": X_train,
        "y_train_original": y_train,
        "scaler": scaler,
        "feature_names": list(X.columns),
    }


if __name__ == "__main__":
    data = run_pipeline()
    print("\nPipeline complete!")
    print(f"Training samples (after SMOTE): {len(data['X_train']):,}")
    print(f"Test samples: {len(data['X_test']):,}")
