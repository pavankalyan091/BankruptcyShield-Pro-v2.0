"""
Prediction Module — v3 (Mapping Fixed)
=====================================================================
Supports: Manual Input, CSV Upload, Company Name (Multi-Source Ticker)

PIPELINE (all 3 modes — same core):
  8 raw numbers
      ↓
  feature_engine.compute_all_features()  → dict of 95 computed ratios
      ↓
  Build DataFrame using bundle['dataset_columns']  ← ORDER FROM TRAINING
  (fills missing features with feature_means)
      ↓
  scaler.transform(df)   ← scaler fit on these exact columns in this exact order
      ↓
  select bundle['selected_idx']  ← GA-chosen feature indices
      ↓
  model.predict_proba()  → Bankrupt / Non-Bankrupt + probability %

KEY FIX: prediction always builds features in the SAME COLUMN ORDER
as training — using bundle['dataset_columns'] instead of a hardcoded list.
This means: re-train on ANY new dataset → prediction still maps correctly.

TICKER RESOLUTION ORDER:
  1. yfinance Search API (global)
  2. NSE India Autocomplete API
  3. Candidate ticker guessing (.NS → .BO → no suffix)

YFINANCE FINANCIAL FETCH:
  Primary  : balance_sheet + income_stmt + cashflow  (yfinance 0.2+)
  Fallback : info dict
"""
import os, sys, json, joblib, warnings, time
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engine import compute_all_features

BASE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, 'models')


# ─── MODEL LOADING ────────────────────────────────────────────────────────────

def load_assets():
    path = os.path.join(MODELS_DIR, 'best_model.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError("No trained model found. Please train first via /train page.")
    bundle = joblib.load(path)

    model          = bundle['model']
    scaler         = bundle['scaler']
    selected_idx   = bundle['selected_idx']
    selected_names = bundle['selected_names']

    # Use saved dataset_columns if available (new bundles), else fallback to
    # scaler feature names attribute, else hardcoded default
    if 'dataset_columns' in bundle:
        dataset_columns = bundle['dataset_columns']
    elif hasattr(scaler, 'feature_names_in_'):
        dataset_columns = list(scaler.feature_names_in_)
    else:
        dataset_columns = _default_feature_names()  # legacy fallback

    means_path    = os.path.join(MODELS_DIR, 'feature_means.pkl')
    feature_means = joblib.load(means_path) if os.path.exists(means_path) else {}
    # Remove target column from means if accidentally saved (legacy bug fix)
    feature_means.pop('Bankrupt?', None)
    feature_means.pop('Bankrupt', None)

    return model, scaler, selected_idx, selected_names, dataset_columns, feature_means


# ─── PIPELINE CORE ────────────────────────────────────────────────────────────

def _raw_to_model_input(raw_data, model, scaler, selected_idx, dataset_columns, feature_means):
    """
    8 raw numbers → 95 computed ratios → align to dataset_columns order
    → QuantileTransformer (maps any scale to uniform [0,1]) → select GA indices.

    OPTION B FIX: Uses QuantileTransformer trained on both original dataset
    AND synthetic extreme bankrupt/healthy samples. This makes the model
    understand raw financial ratios (negative ROA, debt > assets) correctly.
    """
    # Step 1: compute all features
    computed = compute_all_features(raw_data)

    # Step 2: build row in EXACT column order scaler expects
    row = []
    for col in dataset_columns:
        val = computed.get(col, feature_means.get(col, 0.0))
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            val = feature_means.get(col, 0.0)
        row.append(float(val))

    # Step 3: QuantileTransform (works with numpy array, not DataFrame)
    X_arr         = np.array([row])
    X_scaled_full = scaler.transform(X_arr)
    return X_scaled_full[:, selected_idx]


# ─── MANUAL PREDICTION ────────────────────────────────────────────────────────

def _get_optimal_threshold(default=0.35):
    """Load optimal threshold from model bundle (saved during training). Fallback to default."""
    try:
        bundle = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
        return bundle.get('optimal_threshold', default)
    except Exception:
        return default


def predict_from_raw(raw_data, threshold=None):
    """
    8 basic financial inputs → predict.
    threshold: if None, uses optimal_threshold saved during training (auto-tuned).
               Pass a float (e.g. 0.30) to override manually.
    Returns: (status_str, probability_percent)
    """
    model, scaler, selected_idx, selected_names, dataset_columns, feature_means = load_assets()
    if threshold is None:
        threshold = _get_optimal_threshold()
    X    = _raw_to_model_input(raw_data, model, scaler, selected_idx, dataset_columns, feature_means)
    prob = float(model.predict_proba(X)[0][1])
    return ("Bankrupt" if prob >= threshold else "Non-Bankrupt"), round(prob * 100, 2)


# ─── CSV PREDICTION ───────────────────────────────────────────────────────────

def predict_csv(file_path_or_buffer, threshold=None):
    """
    CSV prediction. Handles two formats:
    (A) 8 basic financial columns  → feature_engine per row
    (B) Full dataset feature columns → scale directly

    Appends 'Prediction' and 'Bankruptcy_Probability_%' columns.
    """
    model, scaler, selected_idx, selected_names, dataset_columns, feature_means = load_assets()
    if threshold is None:
        threshold = _get_optimal_threshold()
    df = pd.read_csv(file_path_or_buffer)
    df.columns = df.columns.str.strip()

    basic_cols    = {'revenue','total_assets','total_liabilities','operating_income',
                     'net_income','op_cash_flow','current_assets','current_liabilities'}
    df_lower_cols = {c.lower().replace(' ','_') for c in df.columns}
    results, probs = [], []

    if bool(basic_cols & df_lower_cols):
        # Format A: basic financial columns → feature engine
        col_map = {c.lower().replace(' ','_'): c for c in df.columns}
        for _, row in df.iterrows():
            raw  = {b: float(row.get(col_map.get(b, b), 0) or 0) for b in basic_cols}
            X    = _raw_to_model_input(raw, model, scaler, selected_idx, dataset_columns, feature_means)
            prob = float(model.predict_proba(X)[0][1])
            results.append("Bankrupt" if prob >= threshold else "Non-Bankrupt")
            probs.append(round(prob * 100, 2))
    else:
        # Format B: dataset-style columns → align to dataset_columns order, scale, predict
        for col in dataset_columns:
            if col not in df.columns:
                df[col] = feature_means.get(col, 0.0)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(feature_means.get(col, 0.0))
        X_full = scaler.transform(df[dataset_columns].values)  # numpy array for QuantileTransformer
        for prob in model.predict_proba(X_full[:, selected_idx])[:, 1]:
            results.append("Bankrupt" if prob >= threshold else "Non-Bankrupt")
            probs.append(round(float(prob) * 100, 2))

    df['Prediction']               = results
    df['Bankruptcy_Probability_%'] = probs
    return df


# ─── YFINANCE FINANCIAL FETCH ─────────────────────────────────────────────────

def _safe_val(df_stmt, *row_keys):
    """Extract first non-null numeric value from a yfinance statement DataFrame."""
    if df_stmt is None or (hasattr(df_stmt, 'empty') and df_stmt.empty):
        return 0.0
    for key in row_keys:
        if key in df_stmt.index:
            row = df_stmt.loc[key]
            val = row.iloc[0] if len(row) > 0 else None
            if val is not None:
                try:
                    f = float(val)
                    if not np.isnan(f):
                        return f
                except (ValueError, TypeError):
                    pass
    return 0.0


def fetch_financials_yfinance(ticker_obj):
    """
    Fetch 8 raw financials from yfinance Ticker.
    Primary: balance_sheet / income_stmt / cashflow (yfinance 0.2+)
    Fallback: .info dict (older yfinance / when statements unavailable)

    Returns: (raw_dict, meta_dict)
    """
    raw = {k: 0.0 for k in ['revenue','total_assets','total_liabilities',
                              'operating_income','net_income','op_cash_flow',
                              'current_assets','current_liabilities']}
    meta = {}

    # ── Primary: financial statement DataFrames ───────────────────────────────
    try:
        bs  = ticker_obj.balance_sheet
        inc = ticker_obj.income_stmt
        cf  = ticker_obj.cashflow

        raw['total_assets']        = _safe_val(bs,
            'Total Assets', 'TotalAssets')
        raw['total_liabilities']   = _safe_val(bs,
            'Total Liabilities Net Minority Interest',
            'Total Liab', 'TotalLiab', 'Total Liabilities', 'TotalLiabilities')
        raw['current_assets']      = _safe_val(bs,
            'Current Assets', 'Total Current Assets', 'TotalCurrentAssets')
        raw['current_liabilities'] = _safe_val(bs,
            'Current Liabilities', 'Total Current Liabilities', 'TotalCurrentLiabilities')
        raw['revenue']             = _safe_val(inc,
            'Total Revenue', 'Revenue', 'TotalRevenue')
        raw['operating_income']    = _safe_val(inc,
            'Operating Income', 'OperatingIncome', 'EBIT', 'Ebit')
        raw['net_income']          = _safe_val(inc,
            'Net Income', 'NetIncome',
            'Net Income Common Stockholders',
            'Net Income From Continuing Operations')
        raw['op_cash_flow']        = _safe_val(cf,
            'Operating Cash Flow', 'OperatingCashFlow',
            'Cash Flow From Continuing Operating Activities',
            'Total Cash From Operating Activities')
    except Exception:
        pass

    # ── Fallback: .info dict ──────────────────────────────────────────────────
    try:
        info = ticker_obj.info or {}
        meta = {
            'longName':          info.get('longName') or info.get('shortName', ''),
            'exchange':          info.get('exchange', ''),
            'sector':            info.get('sector', 'N/A'),
            'industry':          info.get('industry', 'N/A'),
            'country':           info.get('country', 'N/A'),
            'financialCurrency': info.get('financialCurrency', 'USD'),
        }
        # Use info dict financials only if statements gave nothing
        if raw['total_assets'] == 0 and raw['revenue'] == 0:
            raw['total_assets']        = float(info.get('totalAssets', 0) or 0)
            raw['total_liabilities']   = float(info.get('totalLiab', 0)
                                               or info.get('totalLiabilities', 0) or 0)
            raw['current_assets']      = float(info.get('totalCurrentAssets', 0) or 0)
            raw['current_liabilities'] = float(info.get('totalCurrentLiabilities', 0) or 0)
            raw['revenue']             = float(info.get('totalRevenue', 0) or 0)
            raw['operating_income']    = float(info.get('operatingIncome', 0) or 0)
            raw['net_income']          = float(info.get('netIncome', 0) or 0)
            raw['op_cash_flow']        = float(info.get('operatingCashflow', 0) or 0)
    except Exception:
        pass

    return raw, meta


# ─── MULTI-SOURCE TICKER RESOLVER ────────────────────────────────────────────

def _try_fetch(symbol):
    """Attempt to get a yfinance Ticker with financial data. Returns (raw, meta) or None."""
    try:
        import yfinance as yf
        t   = yf.Ticker(symbol)
        raw, meta = fetch_financials_yfinance(t)
        if raw['total_assets'] != 0 or raw['revenue'] != 0:
            if not meta.get('longName'):
                meta['longName'] = symbol
            return raw, meta
    except Exception:
        pass
    return None


def _try_nse_api(company_name):
    """NSE India autocomplete → list of base symbols."""
    try:
        import requests
        sess = requests.Session()
        sess.get('https://www.nseindia.com', timeout=6,
                 headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        time.sleep(0.4)
        r = sess.get(
            f'https://www.nseindia.com/api/search-autocomplete?q={company_name}',
            headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json',
                     'Referer': 'https://www.nseindia.com/'},
            timeout=8)
        if r.status_code == 200:
            return [i.get('symbol', '') for i in r.json().get('data', [])[:5] if i.get('symbol')]
    except Exception:
        pass
    return []


def _candidate_tickers(company_name):
    """Generate plausible ticker symbols from company name."""
    STRIP = {'LTD','LIMITED','INC','INDUSTRIES','INDUSTRY','CORP','CORPORATION',
             'PVT','PRIVATE','GROUP','HOLDINGS','HOLDING','VENTURES','ENTERPRISES',
             'TECHNOLOGIES','TECHNOLOGY','SOLUTIONS','SERVICES','INDIA',
             'INTERNATIONAL','GLOBAL','FINANCE','FINANCIAL','AND','THE','OF'}
    words = [w for w in company_name.upper().split() if len(w) > 1]
    core  = [w for w in words if w not in STRIP] or words

    seen, out = set(), []
    def add(c):
        if c and c not in seen:
            seen.add(c); out.append(c)

    if core:
        add(core[0]); add(core[0][:6]); add(core[0][:5]); add(core[0][:4])
    if len(core) >= 2:
        add(''.join(w[0] for w in core))
        add(core[0][:5] + core[1][0])
        add(core[0] + core[1][:3])
    if len(core) >= 3:
        add(''.join(w[0] for w in core[:3]))
    return out


def resolve_ticker_multisource(company_name):
    """
    Company name → (symbol, raw_financials, meta, exchange_label)
    Tries: yfinance Search → NSE API → ticker guessing (.NS / .BO / none)
    Returns (None, None, None, None) if all sources fail.
    """
    import yfinance as yf

    # 1. yfinance Search
    try:
        quotes = getattr(yf.Search(company_name, max_results=10), 'quotes', []) or []
        for q in quotes:
            sym = q.get('symbol', '').strip()
            if not sym:
                continue
            result = _try_fetch(sym)
            if result:
                raw, meta = result
                exch = meta.get('exchange', '') or q.get('exchange', 'Yahoo Finance')
                return sym, raw, meta, f"Yahoo Finance ({exch})"
    except Exception:
        pass

    # 2. NSE India API
    for sym in _try_nse_api(company_name):
        for suffix, label in [('.NS', 'NSE India'), ('.BO', 'BSE India')]:
            result = _try_fetch(sym + suffix)
            if result:
                raw, meta = result
                return sym + suffix, raw, meta, label

    # 3. Candidate ticker guessing
    for base in _candidate_tickers(company_name):
        for suffix, label in [('.NS', 'NSE India'), ('.BO', 'BSE India'), ('', 'Global Exchange')]:
            result = _try_fetch(base + suffix)
            if result:
                raw, meta = result
                return base + suffix, raw, meta, label

    return None, None, None, None


# ─── COMPANY NAME PREDICTION ──────────────────────────────────────────────────

def predict_company(company_name, threshold=None):
    """
    Resolves company name → ticker → yfinance financials → predict.
    Same internal pipeline as Manual Input.

    Returns dict with: company, symbol, exchange, status, probability,
                       financial_data, currency, sector, industry, country
    OR {'error': '...'} on failure.
    """
    try:
        import yfinance as yf  # validate import

        symbol, raw, meta, exchange = resolve_ticker_multisource(company_name)

        if not symbol or not raw:
            return {'error': (
                f'Could not resolve ticker for "{company_name}". '
                'Tried: Yahoo Finance Search, NSE India API, BSE India. '
                'Tip: Use full listed name, e.g. "Infosys Limited", "Reliance Industries Ltd".'
            )}

        if raw['total_assets'] == 0 and raw['revenue'] == 0:
            return {'error': (
                f'Ticker {symbol} found ({exchange}) but no financial data available. '
                'Company may be delisted, too small, or data is private.'
            )}

        status, prob = predict_from_raw(raw, threshold)

        return {
            'company':        meta.get('longName', company_name),
            'input_name':     company_name,
            'symbol':         symbol,
            'exchange':       exchange,
            'status':         status,
            'probability':    prob,
            'currency':       meta.get('financialCurrency', 'USD'),
            'sector':         meta.get('sector', 'N/A'),
            'industry':       meta.get('industry', 'N/A'),
            'country':        meta.get('country', 'N/A'),
            'financial_data': {
                'Total Revenue':       raw['revenue'],
                'Total Assets':        raw['total_assets'],
                'Total Liabilities':   raw['total_liabilities'],
                'Operating Income':    raw['operating_income'],
                'Net Income':          raw['net_income'],
                'Operating Cash Flow': raw['op_cash_flow'],
                'Current Assets':      raw['current_assets'],
                'Current Liabilities': raw['current_liabilities'],
            },
        }

    except ImportError:
        return {'error': 'yfinance not installed. Run: pip install yfinance'}
    except Exception as e:
        return {'error': f'Unexpected error: {str(e)}'}


# ─── LEGACY FALLBACK (if bundle has no dataset_columns) ──────────────────────

def _default_feature_names():
    """Default 95 feature names for Taiwan bankruptcy dataset (legacy fallback only)."""
    return [
        'ROA(C) before interest and depreciation before interest',
        'ROA(A) before interest and % after tax',
        'ROA(B) before interest and depreciation after tax',
        'Operating Gross Margin','Realized Sales Gross Margin','Operating Profit Rate',
        'Pre-tax net Interest Rate','After-tax net Interest Rate',
        'Non-industry income and expenditure/revenue','Continuous interest rate (after tax)',
        'Operating Expense Rate','Research and development expense rate','Cash flow rate',
        'Interest-bearing debt interest rate','Tax rate (A)',
        'Net Value Per Share (B)','Net Value Per Share (A)','Net Value Per Share (C)',
        'Persistent EPS in the Last Four Seasons','Cash Flow Per Share',
        'Revenue Per Share (Yuan ¥)','Operating Profit Per Share (Yuan ¥)',
        'Per Share Net profit before tax (Yuan ¥)',
        'Realized Sales Gross Profit Growth Rate','Operating Profit Growth Rate',
        'After-tax Net Profit Growth Rate','Regular Net Profit Growth Rate',
        'Continuous Net Profit Growth Rate','Total Asset Growth Rate','Net Value Growth Rate',
        'Total Asset Return Growth Rate Ratio','Cash Reinvestment %',
        'Current Ratio','Quick Ratio','Interest Expense Ratio',
        'Total debt/Total net worth','Debt ratio %','Net worth/Assets',
        'Long-term fund suitability ratio (A)','Borrowing dependency',
        'Contingent liabilities/Net worth','Operating profit/Paid-in capital',
        'Net profit before tax/Paid-in capital','Inventory and accounts receivable/Net value',
        'Total Asset Turnover','Accounts Receivable Turnover','Average Collection Days',
        'Inventory Turnover Rate (times)','Fixed Assets Turnover Frequency',
        'Net Worth Turnover Rate (times)','Revenue per person','Operating profit per person',
        'Allocation rate per person','Working Capital to Total Assets',
        'Quick Assets/Total Assets','Current Assets/Total Assets','Cash/Total Assets',
        'Quick Assets/Current Liability','Cash/Current Liability','Current Liability to Assets',
        'Operating Funds to Liability','Inventory/Working Capital','Inventory/Current Liability',
        'Current Liabilities/Liability','Working Capital/Equity','Current Liabilities/Equity',
        'Long-term Liability to Current Assets','Retained Earnings to Total Assets',
        'Total income/Total expense','Total expense/Assets',
        'Current Asset Turnover Rate','Quick Asset Turnover Rate',
        'Working capitcal Turnover Rate','Cash Turnover Rate','Cash Flow to Sales',
        'Fixed Assets to Assets','Current Liability to Liability','Current Liability to Equity',
        'Equity to Long-term Liability','Cash Flow to Total Assets','Cash Flow to Liability',
        'CFO to Assets','Cash Flow to Equity','Current Liability to Current Assets',
        'Liability-Assets Flag','Net Income to Total Assets','Total assets to GNP price',
        'No-credit Interval','Gross Profit to Sales',"Net Income to Stockholder's Equity",
        'Liability to Equity','Degree of Financial Leverage (DFL)',
        'Interest Coverage Ratio (Interest expense to EBIT)','Net Income Flag','Equity to Liability',
    ]


if __name__ == "__main__":
    for name in ["Infosys", "Reliance Industries", "Apple", "Tesla", "Tata Motors"]:
        print(f"\n[{name}]:", predict_company(name))
