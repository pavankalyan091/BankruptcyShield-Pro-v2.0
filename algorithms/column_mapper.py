"""
Column Mapper Module
====================
Converts ANY user CSV into the standard format our system needs.

Two Modes:
  Mode A — Basic (8 columns): User provides raw financial numbers.
            We run feature_engine to compute 95 ratios.
  Mode B — Full (95+ columns): User provides full feature dataset (Taiwan format).
            We use columns directly.

Auto-detection + fuzzy matching for common financial column name variations.
"""

import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── The 8 standard fields we need for Mode A ──────────────────────────────────
REQUIRED_FIELDS = {
    'revenue':             'Total Revenue / Net Sales',
    'total_assets':        'Total Assets',
    'total_liabilities':   'Total Liabilities',
    'operating_income':    'Operating Income / EBIT',
    'net_income':          'Net Income / Net Profit',
    'op_cash_flow':        'Operating Cash Flow',
    'current_assets':      'Current Assets',
    'current_liabilities': 'Current Liabilities',
}

# Fuzzy keyword patterns for auto-detection
FIELD_KEYWORDS = {
    'revenue':             ['revenue', 'sales', 'net sales', 'turnover', 'income from operations', 'total revenue'],
    'total_assets':        ['total assets', 'total_assets', 'assets total', 'sum of assets'],
    'total_liabilities':   ['total liabilities', 'total_liabilities', 'liabilities total', 'total debt', 'total_debt'],
    'operating_income':    ['operating income', 'operating_income', 'ebit', 'operating profit', 'income from operations'],
    'net_income':          ['net income', 'net_income', 'net profit', 'profit after tax', 'pat', 'net earnings'],
    'op_cash_flow':        ['operating cash', 'cash from operations', 'operating_cash_flow', 'op_cash_flow', 'cfo'],
    'current_assets':      ['current assets', 'current_assets', 'short term assets'],
    'current_liabilities': ['current liabilities', 'current_liabilities', 'short term liabilities', 'short-term liabilities'],
}

TARGET_KEYWORDS = ['bankrupt', 'bankruptcy', 'default', 'target', 'label', 'status', 'failed', 'insolvent']


def detect_csv_mode(df):
    """
    Returns:
      'full'  — dataset has 95+ numeric columns (Taiwan-style, train directly)
      'basic' — dataset has ~8 financial columns (need feature_engine)
      'unknown' — can't determine
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove likely target columns
    feature_cols = [c for c in numeric_cols if not any(t in c.lower() for t in TARGET_KEYWORDS)]

    if len(feature_cols) >= 50:
        return 'full'
    elif len(feature_cols) >= 3:
        return 'basic'
    return 'unknown'


def auto_map_columns(df_columns):
    """
    Auto-detect which CSV column maps to which of our 8 fields.
    Returns dict: {our_field: csv_column or None}
    """
    cols_lower = {c.lower().strip(): c for c in df_columns}
    mapping = {}

    for field, keywords in FIELD_KEYWORDS.items():
        matched = None
        for kw in keywords:
            if kw in cols_lower:
                matched = cols_lower[kw]
                break
        # Try partial match if exact not found
        if not matched:
            for csv_col_lower, csv_col in cols_lower.items():
                for kw in keywords:
                    if kw in csv_col_lower or csv_col_lower in kw:
                        matched = csv_col
                        break
                if matched:
                    break
        mapping[field] = matched

    return mapping


def detect_target_column(df):
    """Auto-detect binary target column."""
    for col in df.columns:
        if any(t in col.lower() for t in TARGET_KEYWORDS):
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                return col
    # Fallback: any binary column
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            return col
    raise ValueError("Cannot detect target column. Need a binary (0/1) column for Bankrupt/Not-Bankrupt.")


def apply_mapping_and_convert(df, column_mapping, target_col):
    """
    Apply user's column mapping, run feature_engine on each row,
    return a full 95-feature DataFrame ready for training.

    column_mapping: {our_field: csv_column_name}
    """
    from feature_engine import compute_all_features

    rows = []
    labels = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            raw = {}
            for field, csv_col in column_mapping.items():
                if csv_col and csv_col in df.columns:
                    val = row[csv_col]
                    raw[field] = float(val) if pd.notna(val) else 0.0
                else:
                    raw[field] = 0.0

            # Basic sanity: total_assets must be > 0
            if raw.get('total_assets', 0) <= 0:
                raw['total_assets'] = 1.0

            computed = compute_all_features(raw)
            rows.append(computed)
            labels.append(int(row[target_col]))
        except Exception:
            skipped += 1
            continue

    if not rows:
        raise ValueError("No valid rows could be processed from the mapped columns.")

    df_features = pd.DataFrame(rows)

    # Fill any NaN with column means
    df_features = df_features.fillna(df_features.mean())
    df_features['Bankrupt?'] = labels

    return df_features, skipped


def get_column_info(df):
    """Return column analysis for the UI."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    try:
        target = detect_target_column(df)
        bankrupt_count = int(df[target].sum())
        total_count = len(df)
    except Exception:
        target = None
        bankrupt_count = 0
        total_count = len(df)

    mode = detect_csv_mode(df)
    auto_mapping = auto_map_columns(df.columns)
    confidence = sum(1 for v in auto_mapping.values() if v is not None)

    return {
        'total_rows': total_count,
        'total_cols': len(all_cols),
        'numeric_cols': numeric_cols,
        'all_cols': all_cols,
        'target_col': target,
        'bankrupt_count': bankrupt_count,
        'healthy_count': total_count - bankrupt_count,
        'mode': mode,
        'auto_mapping': auto_mapping,
        'mapping_confidence': confidence,
    }
