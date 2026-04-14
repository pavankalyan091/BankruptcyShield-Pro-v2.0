"""
Feature Engineering Engine
Accepts 8 basic financial inputs and calculates ALL 95 dataset financial ratios.
Then maps to GA-selected features for model input.
"""

import numpy as np
import json
import joblib
import os

def _safe_div(a, b, default=0.0):
    try:
        if b == 0 or b is None:
            return default
        result = a / b
        if np.isnan(result) or np.isinf(result):
            return default
        return float(result)
    except:
        return default


def compute_all_features(raw_data):
    """
    Accepts a dict with keys: revenue, total_assets, total_liabilities,
    operating_income, net_income, op_cash_flow, current_assets, current_liabilities
    Returns a dict mapping ALL 95 dataset feature names to computed values.
    """
    rev  = float(raw_data.get('revenue', 0) or 0)
    ta   = float(raw_data.get('total_assets', 1) or 1)
    tl   = float(raw_data.get('total_liabilities', 0) or 0)
    oi   = float(raw_data.get('operating_income', 0) or 0)
    ni   = float(raw_data.get('net_income', 0) or 0)
    ocf  = float(raw_data.get('op_cash_flow', 0) or 0)
    ca   = float(raw_data.get('current_assets', 0) or 0)
    cl   = float(raw_data.get('current_liabilities', 0) or 0)

    equity  = ta - tl
    wc      = ca - cl
    lt_liab = max(tl - cl, 0)
    fa      = max(ta - ca, 0)
    expense = max(rev - oi, 0.01)

    f = {}
    f['ROA(C) before interest and depreciation before interest'] = _safe_div(oi, ta)
    f['ROA(A) before interest and % after tax']                  = _safe_div(ni, ta)
    f['ROA(B) before interest and depreciation after tax']       = _safe_div(ni + oi*0.1, ta)
    f['Operating Gross Margin']                                  = _safe_div(oi, rev)
    f['Realized Sales Gross Margin']                             = _safe_div(oi, rev)
    f['Operating Profit Rate']                                   = _safe_div(oi, rev)
    f['Pre-tax net Interest Rate']                               = _safe_div(ni, rev)
    f['After-tax net Interest Rate']                             = _safe_div(ni, rev)
    f['Non-industry income and expenditure/revenue']             = 0.0   # no proxy available
    f['Continuous interest rate (after tax)']                    = _safe_div(ni, ta)
    f['Operating Expense Rate']                                  = _safe_div(expense, rev)
    f['Research and development expense rate']                   = 0.0   # not in basic inputs; 0 is correct default
    f['Cash flow rate']                                          = _safe_div(ocf, rev)
    f['Interest-bearing debt interest rate']                     = _safe_div(tl, ta)  # proxy: debt burden ratio
    f['Tax rate (A)']                                            = 0.25 if ni > 0 else 0.0  # proxy: profitable→25% else 0
    f['Net Value Per Share (B)']                                 = _safe_div(equity, ta)
    f['Net Value Per Share (A)']                                 = _safe_div(equity, ta)
    f['Net Value Per Share (C)']                                 = _safe_div(equity, ta)
    f['Persistent EPS in the Last Four Seasons']                 = _safe_div(ni, ta)
    f['Cash Flow Per Share']                                     = _safe_div(ocf, ta)
    f['Revenue Per Share (Yuan ¥)']                              = _safe_div(rev, ta)
    f['Operating Profit Per Share (Yuan ¥)']                     = _safe_div(oi, ta)
    f['Per Share Net profit before tax (Yuan ¥)']                = _safe_div(ni, ta)
    f['Realized Sales Gross Profit Growth Rate']                 = 0.0   # needs historical data
    f['Operating Profit Growth Rate']                            = 0.0   # needs historical data
    f['After-tax Net Profit Growth Rate']                        = 0.0   # needs historical data
    f['Regular Net Profit Growth Rate']                          = 0.0   # needs historical data
    f['Continuous Net Profit Growth Rate']                       = 0.0   # needs historical data
    f['Total Asset Growth Rate']                                 = 0.0   # needs historical data
    f['Net Value Growth Rate']                                   = _safe_div(ni, equity) if equity > 0 else 0.0
    f['Total Asset Return Growth Rate Ratio']                    = 0.0   # needs historical data
    f['Cash Reinvestment %']                                     = _safe_div(ocf, equity) if equity > 0 else 0.0
    f['Current Ratio']                                           = _safe_div(ca, cl)
    f['Quick Ratio']                                             = _safe_div(ca * 0.8, cl)
    f['Interest Expense Ratio']                                  = _safe_div(tl, ta)  # proxy: total debt / total assets
    f['Total debt/Total net worth']                              = _safe_div(tl, equity) if equity > 0 else 10.0
    f['Debt ratio %']                                            = _safe_div(tl, ta)
    f['Net worth/Assets']                                        = _safe_div(equity, ta)
    f['Long-term fund suitability ratio (A)']                    = _safe_div(equity + lt_liab, fa) if fa > 0 else 1.0
    f['Borrowing dependency']                                    = _safe_div(tl, ta)
    f['Contingent liabilities/Net worth']                        = _safe_div(tl, equity) if equity > 0 else 10.0
    f['Operating profit/Paid-in capital']                        = _safe_div(oi, equity) if equity > 0 else 0.0
    f['Net profit before tax/Paid-in capital']                   = _safe_div(ni, equity) if equity > 0 else 0.0
    f['Inventory and accounts receivable/Net value']             = 0.0
    f['Total Asset Turnover']                                    = _safe_div(rev, ta)
    f['Accounts Receivable Turnover']                            = _safe_div(rev, ca * 0.3) if ca > 0 else 0.0
    f['Average Collection Days']                                 = _safe_div(ta * 365, rev) if rev > 0 else 0.0
    f['Inventory Turnover Rate (times)']                         = _safe_div(rev, ca * 0.3) if ca > 0 else 0.0
    f['Fixed Assets Turnover Frequency']                         = _safe_div(rev, fa) if fa > 0 else 0.0
    f['Net Worth Turnover Rate (times)']                         = _safe_div(rev, equity) if equity > 0 else 0.0
    f['Revenue per person']                                      = _safe_div(rev, ta)
    f['Operating profit per person']                             = _safe_div(oi, ta)
    f['Allocation rate per person']                              = _safe_div(ni, ta)
    f['Working Capital to Total Assets']                         = _safe_div(wc, ta)
    f['Quick Assets/Total Assets']                               = _safe_div(ca * 0.8, ta)
    f['Current Assets/Total Assets']                             = _safe_div(ca, ta)
    f['Cash/Total Assets']                                       = _safe_div(ocf, ta)
    f['Quick Assets/Current Liability']                          = _safe_div(ca * 0.8, cl)
    f['Cash/Current Liability']                                  = _safe_div(ocf, cl)
    f['Current Liability to Assets']                             = _safe_div(cl, ta)
    f['Operating Funds to Liability']                            = _safe_div(oi, tl) if tl > 0 else 0.0
    f['Inventory/Working Capital']                               = 0.0
    f['Inventory/Current Liability']                             = 0.0
    f['Current Liabilities/Liability']                           = _safe_div(cl, tl) if tl > 0 else 0.0
    f['Working Capital/Equity']                                  = _safe_div(wc, equity) if equity > 0 else 0.0
    f['Current Liabilities/Equity']                              = _safe_div(cl, equity) if equity > 0 else 0.0
    f['Long-term Liability to Current Assets']                   = _safe_div(lt_liab, ca) if ca > 0 else 0.0
    f['Retained Earnings to Total Assets']                       = _safe_div(ni, ta)
    f['Total income/Total expense']                              = _safe_div(rev, expense)
    f['Total expense/Assets']                                    = _safe_div(expense, ta)
    f['Current Asset Turnover Rate']                             = _safe_div(rev, ca) if ca > 0 else 0.0
    f['Quick Asset Turnover Rate']                               = _safe_div(rev, ca * 0.8) if ca > 0 else 0.0
    f['Working capitcal Turnover Rate']                          = _safe_div(rev, wc) if wc != 0 else 0.0
    f['Cash Turnover Rate']                                      = _safe_div(rev, ocf) if ocf != 0 else 0.0
    f['Cash Flow to Sales']                                      = _safe_div(ocf, rev)
    f['Fixed Assets to Assets']                                  = _safe_div(fa, ta)
    f['Current Liability to Liability']                          = _safe_div(cl, tl) if tl > 0 else 0.0
    f['Current Liability to Equity']                             = _safe_div(cl, equity) if equity > 0 else 0.0
    f['Equity to Long-term Liability']                           = _safe_div(equity, lt_liab) if lt_liab > 0 else 10.0
    f['Cash Flow to Total Assets']                               = _safe_div(ocf, ta)
    f['Cash Flow to Liability']                                  = _safe_div(ocf, tl) if tl > 0 else 0.0
    f['CFO to Assets']                                           = _safe_div(ocf, ta)
    f['Cash Flow to Equity']                                     = _safe_div(ocf, equity) if equity > 0 else 0.0
    f['Current Liability to Current Assets']                     = _safe_div(cl, ca) if ca > 0 else 0.0
    f['Liability-Assets Flag']                                   = 1.0 if tl > ta else 0.0
    f['Net Income to Total Assets']                              = _safe_div(ni, ta)
    f['Total assets to GNP price']                               = 0.0
    f['No-credit Interval']                                      = 0.0
    f['Gross Profit to Sales']                                   = _safe_div(oi, rev)
    f["Net Income to Stockholder's Equity"]                      = _safe_div(ni, equity) if equity > 0 else 0.0
    f['Liability to Equity']                                     = _safe_div(tl, equity) if equity > 0 else 10.0
    f['Degree of Financial Leverage (DFL)']                      = _safe_div(oi, ni) if ni != 0 else 1.0  # proxy
    f['Interest Coverage Ratio (Interest expense to EBIT)']      = _safe_div(tl * 0.05, oi) if oi != 0 else 0.0  # proxy: ~5% interest rate
    f['Net Income Flag']                                         = 1.0 if ni > 0 else 0.0
    f['Equity to Liability']                                     = _safe_div(equity, tl) if tl > 0 else 10.0
    return f


def get_model_input_vector(raw_data, selected_features=None, feature_means=None):
    """
    Full pipeline: raw_data -> compute_all_features -> select GA features -> return vector
    """
    base = os.path.dirname(os.path.abspath(__file__))

    if selected_features is None:
        json_path = os.path.join(base, 'models', 'selected_features.json')
        with open(json_path, 'r') as f:
            selected_features = json.load(f)

    if feature_means is None:
        means_path = os.path.join(base, 'models', 'feature_means.pkl')
        if os.path.exists(means_path):
            feature_means = joblib.load(means_path)
        else:
            feature_means = {}

    all_features = compute_all_features(raw_data)

    vector = []
    for feat in selected_features:
        val = all_features.get(feat, feature_means.get(feat, 0.0))
        vector.append(val)

    return vector
