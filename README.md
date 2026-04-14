# 🛡️ Bankruptcy Shield – Enhanced

A hybrid ML bankruptcy prediction system using Genetic Algorithm + PSO optimization.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000

## 🔮 Prediction Modes

### 1. Manual Input
Enter 8 basic financial figures — the system computes all 95 ratios automatically.

### 2. CSV Upload
Upload a CSV with either:
- 8 basic columns: `revenue, total_assets, total_liabilities, operating_income, net_income, op_cash_flow, current_assets, current_liabilities`
- Or the full 95 dataset feature columns

### 3. Company Lookup (Multi-Source Ticker Resolution)
Type a company name. The system resolves the ticker using:
1. **Yahoo Finance Search API** (global — US, EU, Asia)
2. **NSE India API** (direct autocomplete endpoint)
3. **Ticker guessing** → tries `.NS` (NSE) then `.BO` (BSE) suffixes
4. **No suffix** (US / other exchanges)

Shows: ticker symbol, exchange, sector, industry, country, currency + financial data used.

## 🧠 ML Pipeline
- Genetic Algorithm → feature selection from 95 features
- PSO Optimization → hyperparameter tuning
- Best of: Random Forest vs Logistic Regression
- Threshold: 30% (high sensitivity for bankruptcy detection)
