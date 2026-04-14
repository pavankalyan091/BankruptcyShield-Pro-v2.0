"""
Flask Web Application - Bankruptcy Prediction System (Final Version)
Features: Column Mapper, Live Training Progress, 3 Prediction Modes
"""
from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify, send_file)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os, sys, json, io, sqlite3, traceback, threading
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.secret_key = "bankruptcy_shield_2025_secure"

BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR  = os.path.join(BASE_DIR, 'uploads')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
DB_PATH     = os.path.join(BASE_DIR, 'database', 'users.db')
STATUS_FILE = os.path.join(MODELS_DIR, 'training_status.json')
MAPPED_DS   = os.path.join(BASE_DIR, 'dataset', 'mapped_dataset.csv')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'database'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'dataset'), exist_ok=True)

_training_lock = threading.Lock()
_training_active = False

# ─────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email    TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        conn.commit()

init_db()
app.jinja_env.globals["enumerate"] = enumerate

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please login first.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapper

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def model_exists():
    return os.path.exists(os.path.join(MODELS_DIR, 'best_model.pkl'))

def load_bundle():
    return joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))

def read_status():
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {'stage': 'IDLE', 'done': False, 'error': None}

# ─────────────────────────────────────────
# PUBLIC ROUTES
# ─────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare')
def compare():
    comparison = None
    comp_path = os.path.join(MODELS_DIR, 'model_comparison.json')
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comparison = json.load(f)
    return render_template('compare.html', comparison=comparison)

# ─────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return render_template('register.html')
        try:
            with get_db() as conn:
                if conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone():
                    flash("Email already registered.", "danger")
                    return render_template('register.html')
                conn.execute("INSERT INTO users (username,email,password) VALUES (?,?,?)",
                             (username, email, generate_password_hash(password)))
                conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"Registration error: {e}", "danger")
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user and check_password_hash(user['password'], password):
            session['user_id']  = user['id']
            session['username'] = user['username']
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for('dashboard'))
        flash("Invalid email or password.", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('index'))

# ─────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    comparison = None
    if model_exists():
        comp_path = os.path.join(MODELS_DIR, 'model_comparison.json')
        if os.path.exists(comp_path):
            with open(comp_path) as f:
                comparison = json.load(f)
    return render_template('dashboard.html', model_trained=model_exists(), comparison=comparison)

# ─────────────────────────────────────────
# TRAIN — with Column Mapper + Live Progress
# ─────────────────────────────────────────

@app.route('/api/analyze_csv', methods=['POST'])
@login_required
def analyze_csv():
    """Analyze uploaded CSV and return column info + auto-mapping suggestions."""
    file = request.files.get('file')
    if not file or not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a valid CSV file.'}), 400
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        from algorithms.column_mapper import get_column_info, detect_csv_mode
        info = get_column_info(df)
        # Save temp file for later use
        tmp_path = os.path.join(UPLOAD_DIR, 'temp_upload.csv')
        df.to_csv(tmp_path, index=False)
        return jsonify({'success': True, 'info': info})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/apply_mapping', methods=['POST'])
@login_required
def apply_mapping():
    """Apply user's column mapping, convert CSV, save as training dataset."""
    tmp_path = os.path.join(UPLOAD_DIR, 'temp_upload.csv')
    if not os.path.exists(tmp_path):
        return jsonify({'error': 'No uploaded file found. Please upload again.'}), 400
    try:
        data = request.get_json()
        mode           = data.get('mode', 'full')        # 'full' or 'basic'
        column_mapping = data.get('mapping', {})         # {our_field: csv_col}
        target_col     = data.get('target_col', '')

        df = pd.read_csv(tmp_path)
        df.columns = df.columns.str.strip()

        ds_path = os.path.join(BASE_DIR, 'dataset', 'bankruptcy.csv')

        if mode == 'full':
            # Taiwan-style: use directly, just rename target if needed
            if target_col and target_col != 'Bankrupt?':
                df = df.rename(columns={target_col: 'Bankrupt?'})
            # Fill NaN
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
            df.to_csv(ds_path, index=False)
            return jsonify({'success': True, 'rows': len(df),
                            'cols': len(df.columns) - 1, 'mode': 'full'})

        elif mode == 'basic':
            # 8-column mode: run feature_engine on each row
            from algorithms.column_mapper import apply_mapping_and_convert, detect_target_column
            if not target_col:
                target_col = detect_target_column(df)
            df_converted, skipped = apply_mapping_and_convert(df, column_mapping, target_col)
            df_converted.to_csv(ds_path, index=False)
            return jsonify({'success': True, 'rows': len(df_converted),
                            'cols': len(df_converted.columns) - 1,
                            'skipped': skipped, 'mode': 'basic'})
        else:
            return jsonify({'error': 'Unknown mode.'}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@app.route('/api/training_status')
@login_required
def training_status():
    """Poll this for real-time training progress."""
    return jsonify(read_status())


@app.route('/api/start_training', methods=['POST'])
@login_required
def start_training_api():
    """Start training in background thread."""
    global _training_active
    with _training_lock:
        if _training_active:
            return jsonify({'error': 'Training already in progress.'}), 400
        _training_active = True

    ds_path = os.path.join(BASE_DIR, 'dataset', 'bankruptcy.csv')
    if not os.path.exists(ds_path):
        _training_active = False
        return jsonify({'error': 'No dataset found. Please upload a dataset first.'}), 400

    def run():
        global _training_active
        try:
            from train_models import train as run_training
            run_training(dataset_path=ds_path)
        except Exception as e:
            import json, traceback
            os.makedirs(MODELS_DIR, exist_ok=True)
            with open(STATUS_FILE, 'w') as f:
                json.dump({'stage': 'ERROR', 'stage_label': 'Training Failed',
                           'step': str(e), 'done': True, 'error': str(e),
                           'gen': 0, 'total_gens': 15, 'best_f1': 0,
                           'features': 0, 'pso_iter': 0, 'total_pso': 8}, f)
            traceback.print_exc()
        finally:
            _training_active = False

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return jsonify({'success': True, 'message': 'Training started!'})


@app.route('/train', methods=['GET', 'POST'])
@login_required
def train():
    ds_path = os.path.join(BASE_DIR, 'dataset', 'bankruptcy.csv')
    dataset_info = None
    if os.path.exists(ds_path):
        try:
            df_info = pd.read_csv(ds_path, nrows=2)
            with open(ds_path) as f:
                n_rows = sum(1 for _ in f) - 1
            dataset_info = {
                'filename': 'bankruptcy.csv',
                'rows': n_rows,
                'columns': len(df_info.columns) - 1
            }
        except Exception:
            dataset_info = {'filename': 'bankruptcy.csv', 'rows': '?', 'columns': '?'}

    status = read_status()
    return render_template('train.html', dataset_info=dataset_info, status=status)

# ─────────────────────────────────────────
# PREDICT — 3 Modes
# ─────────────────────────────────────────
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if not model_exists():
        flash("No trained model found. Please train the model first.", "warning")
        return redirect(url_for('train'))

    result = None
    if request.method == 'POST':
        mode = request.form.get('mode', 'manual')

        # ── MANUAL INPUT ──────────────────────────────────────────────────────
        if mode == 'manual':
            try:
                raw = {
                    'revenue':             float(request.form.get('revenue', 0) or 0),
                    'total_assets':        float(request.form.get('total_assets', 1) or 1),
                    'total_liabilities':   float(request.form.get('total_liabilities', 0) or 0),
                    'operating_income':    float(request.form.get('operating_income', 0) or 0),
                    'net_income':          float(request.form.get('net_income', 0) or 0),
                    'op_cash_flow':        float(request.form.get('op_cash_flow', 0) or 0),
                    'current_assets':      float(request.form.get('current_assets', 0) or 0),
                    'current_liabilities': float(request.form.get('current_liabilities', 0) or 0),
                }
                from predict import predict_from_raw
                status_pred, prob = predict_from_raw(raw)
                try:
                    display_threshold = load_bundle().get('optimal_threshold', 0.35) * 100
                except Exception:
                    display_threshold = 35
                result = {'mode': 'Manual Input', 'status': status_pred,
                          'probability': prob, 'threshold': display_threshold}
            except Exception as e:
                flash(f"Prediction error: {str(e)}", "danger")
                traceback.print_exc()

        # ── CSV UPLOAD ────────────────────────────────────────────────────────
        elif mode == 'csv':
            file = request.files.get('csv_file')
            if not file or not allowed_file(file.filename):
                flash("Please upload a valid CSV file.", "danger")
            else:
                try:
                    from predict import predict_csv
                    df_result = predict_csv(file)
                    out_path = os.path.join(UPLOAD_DIR, 'prediction_results.csv')
                    df_result.to_csv(out_path, index=False)
                    bankrupt_count     = (df_result['Prediction'] == 'Bankrupt').sum()
                    non_bankrupt_count = (df_result['Prediction'] == 'Non-Bankrupt').sum()
                    result = {
                        'mode': 'CSV Upload',
                        'total': len(df_result),
                        'bankrupt': int(bankrupt_count),
                        'non_bankrupt': int(non_bankrupt_count),
                        'preview': df_result[['Prediction', 'Bankruptcy_Probability_%']].head(10).to_dict('records'),
                        'download': True,
                    }
                    flash(f"CSV processed: {len(df_result)} companies analysed.", "success")
                except Exception as e:
                    flash(f"CSV error: {str(e)}", "danger")
                    traceback.print_exc()

        # ── COMPANY NAME ──────────────────────────────────────────────────────
        elif mode == 'company':
            company_name = request.form.get('company_name', '').strip()
            if not company_name:
                flash("Please enter a company name.", "danger")
            else:
                try:
                    from predict import predict_company
                    res = predict_company(company_name)
                    if 'error' in res:
                        flash(f"Company lookup error: {res['error']}", "danger")
                    else:
                        try:
                            display_threshold = load_bundle().get('optimal_threshold', 0.35) * 100
                        except Exception:
                            display_threshold = 35
                        result = {
                            'mode': 'Company Lookup',
                            'company':   res.get('company', company_name),
                            'symbol':    res.get('symbol', ''),
                            'exchange':  res.get('exchange', 'Unknown'),
                            'status':    res['status'],
                            'probability': res['probability'],
                            'threshold': display_threshold,
                            'financial_data': res.get('financial_data', {}),
                            'currency':  res.get('currency', 'USD'),
                            'sector':    res.get('sector', 'N/A'),
                            'industry':  res.get('industry', 'N/A'),
                            'country':   res.get('country', 'N/A'),
                        }
                except Exception as e:
                    flash(f"Company prediction error: {str(e)}", "danger")

    return render_template('predict.html', result=result)


@app.route('/download_results')
@login_required
def download_results():
    out_path = os.path.join(UPLOAD_DIR, 'prediction_results.csv')
    if os.path.exists(out_path):
        return send_file(out_path, as_attachment=True, download_name='bankruptcy_predictions.csv')
    flash("No results file found.", "danger")
    return redirect(url_for('predict'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
