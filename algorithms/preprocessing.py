"""
Preprocessing Pipeline
Reads dataset, fills NaN, splits, scales.
Returns X_train, X_test, y_train, y_test, scaler, feature_names (in CSV order)
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_CANDIDATES = ['Bankrupt?', 'Bankrupt', 'bankrupt', 'target', 'Target',
                     'label', 'Label', 'default', 'Default', 'status', 'Status']

def detect_target_column(df):
    """Auto-detect the binary target column."""
    # Try known names first
    for name in TARGET_CANDIDATES:
        if name in df.columns:
            return name
    # Fallback: find a binary column (only 0/1 values)
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            return col
    raise ValueError(
        f"Cannot detect target column. Expected one of {TARGET_CANDIDATES} "
        "or any column with only 0/1 values."
    )

def run_preprocessing(dataset_path=None):
    if dataset_path is None:
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dataset", "bankruptcy.csv"
        )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()

    # Auto-detect target column
    target_col = detect_target_column(df)

    # Fill NaN with column means (numeric only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df = df.drop_duplicates()

    X = df.drop(target_col, axis=1).select_dtypes(include=[np.number])
    y = df[target_col].astype(int)

    feature_names = list(X.columns)  # Exact CSV column order — critical for scaler

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
