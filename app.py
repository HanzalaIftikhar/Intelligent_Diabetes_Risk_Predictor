from flask import Flask, render_template, request
import joblib
import numpy as np
from fpdf import FPDF
import sqlite3
from datetime import datetime
from flask import session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv

# Absolute path of the project folder
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Important project file paths
DB_PATH = os.path.join(BASE_DIR, 'diabetes.db')
MODEL_PATH = os.path.join(BASE_DIR, 'diabetes_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# Load environment variables from .env file if it exists
load_dotenv(os.path.join(BASE_DIR, '.env'))

app = Flask(__name__)

# Secret key: reads from .env file in production.
# Falls back to a development-only default if .env is not present.
# IMPORTANT: Always set a strong SECRET_KEY in your .env file before deploying.
app.secret_key = os.environ.get('SECRET_KEY', 'dev-fallback-key-change-in-production')

from functools import wraps

def patient_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Not logged in at all — send to login page
        if 'user_id' not in session:
            return redirect(url_for('login'))
        # Logged in but wrong role — show access denied page
        if session.get('user_role') != 'patient':
            return render_template('access_denied.html'), 403
        return f(*args, **kwargs)
    return decorated

def doctor_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Not logged in at all — send to login page
        if 'user_id' not in session:
            return redirect(url_for('login'))
        # Logged in but wrong role — show access denied page
        if session.get('user_role') != 'doctor':
            return render_template('access_denied.html'), 403
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Not logged in at all — send to login page
        if 'user_id' not in session:
            return redirect(url_for('login'))
        # Logged in but wrong role — show access denied page
        if session.get('user_role') != 'admin':
            return render_template('access_denied.html'), 403
        return f(*args, **kwargs)
    return decorated

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Predictions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            preg REAL, plas REAL, pres REAL,
            mass REAL, pedi REAL, age REAL,
            risk TEXT, message TEXT,
            date TEXT
        )
    ''')
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            date TEXT
        )
    ''')
    
    # Feedback table
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            rating INTEGER,
            comment TEXT,
            date TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# -----------------------------------------------------------------------
# INPUT VALIDATION FUNCTION — Task B
# Called inside /predict before any ML code runs.
# Returns (is_valid: bool, error_message: str)
# All limits are based on realistic human medical values.
# -----------------------------------------------------------------------
def validate_prediction_input(form):
    """
    Validates raw form data from the prediction form.
    Returns a tuple: (True, '') if valid, or (False, 'error message') if not.
    """

    # Step 1: Check all fields exist and are numeric
    fields = {
        'preg': 'Pregnancies',
        'plas': 'Glucose Level',
        'pres': 'Blood Pressure',
        'mass': 'BMI',
        'pedi': 'Diabetes Pedigree Function',
        'age':  'Age',
    }

    values = {}
    for field_name, label in fields.items():
        raw = form.get(field_name, '').strip()
        if raw == '':
            return False, f'{label} is required.'
        try:
            values[field_name] = float(raw)
        except ValueError:
            return False, f'{label} must be a number. You entered: "{raw}"'

    
    preg = values['preg']
    plas = values['plas']
    pres = values['pres']
    mass = values['mass']
    pedi = values['pedi']
    age  = values['age']

    # Step 2: Integer-only checks for fields that should not be decimal
    if not preg.is_integer():
        return False, 'Pregnancies must be a whole number.'

    if not age.is_integer():
        return False, 'Age must be a whole number.'

    # Step 3: Range checks

    # Pregnancies: 0 is valid (not pregnant), max 20 is realistic
    if preg < 0 or preg > 20:
        return False, 'Pregnancies must be between 0 and 20.'

    # Glucose: must be positive, realistic range up to 600 mg/dL
    if plas <= 0:
        return False, 'Glucose Level must be greater than 0.'
    if plas > 600:
        return False, 'Glucose Level seems too high (max 600). Please check your value.'

    # Blood Pressure: must be positive, realistic range up to 200 mmHg
    if pres <= 0:
        return False, 'Blood Pressure must be greater than 0.'
    if pres > 200:
        return False, 'Blood Pressure seems too high (max 200). Please check your value.'

    # BMI: must be positive, realistic range up to 100
    if mass <= 0:
        return False, 'BMI must be greater than 0.'
    if mass > 100:
        return False, 'BMI seems too high (max 100). Please check your value.'

    # Diabetes Pedigree Function: 0 is valid, realistic range 0 to 4
    if pedi < 0:
        return False, 'Diabetes Pedigree Function cannot be negative.'
    if pedi > 4:
        return False, 'Diabetes Pedigree Function seems too high (max 4). Please check your value.'

    # Age: must be positive, realistic range up to 120
    if age <= 0:
        return False, 'Age must be greater than 0.'
    if age > 120:
        return False, 'Age seems too high (max 120). Please check your value.'

    # All checks passed
    return True, ''


@app.route('/') 
@patient_required
def home():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['user_name'] = user[1]
            session['user_role'] = user[4]
            
            if user[4] == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user[4] == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            else:
                return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        role = 'patient'  # Hardcoded - sirf patients khud register kar sakte hain
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute('''
                INSERT INTO users (name, email, password, role, date)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, email, password, role, datetime.now().strftime("%Y-%m-%d %H:%M")))
            conn.commit()
            conn.close()
            flash('Account created successfully! Please login.', 'success')

            return redirect(url_for('login'))
        except:
            conn.close()
            flash('Email already exists try another.', 'error')
            return redirect(url_for('register'))
    return render_template('register.html')

@app.route('/predict', methods=['POST'])
@patient_required
def predict():

    # Validate input before calling the ML model
    is_valid, error_message = validate_prediction_input(request.form)
    if not is_valid:
        flash(error_message, 'error')
        return redirect(url_for('home'))

    # Get form data
    preg = float(request.form['preg'])
    plas = float(request.form['plas'])
    pres = float(request.form['pres'])
    mass = float(request.form['mass'])
    pedi = float(request.form['pedi'])
    age = float(request.form['age'])

    # Prepare input
    # Feature order: preg, plas, pres, mass, pedi, age — must not change
    input_data = np.array([[preg, plas, pres, mass, pedi, age]])
    input_scaled = scaler.transform(input_data)

    # Predict
    result = model.predict(input_scaled)[0]

    # Risk level
    if result == 0:
        risk = "Low"
        message = "Patient is likely Non-Diabetic"
    elif plas > 140 or mass > 35:
        risk = "High"
        message = "Patient is likely Diabetic"
    else:
        risk = "Medium"
        message = "Patient may be Diabetic"
    # Save to database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    user_id = session.get('user_id', None)

    c.execute('''
        INSERT INTO predictions (user_id, preg, plas, pres, mass, pedi, age, risk, message, date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, preg, plas, pres, mass, pedi, age, risk, message, datetime.now().strftime("%Y-%m-%d %H:%M")))
    pred_id = c.lastrowid
    conn.commit()
    conn.close()

    return render_template('result.html',
                     risk=risk,
                     message=message,
                     preg=preg, plas=plas, pres=pres,
                     mass=mass, pedi=pedi, age=age,
                     pred_id=pred_id)


@app.route('/dashboard')
@patient_required
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))   
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    user_id = session.get('user_id')
    c.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC', (user_id,))
    predictions = c.fetchall()
    conn.close()
    
    # Risk counts
    low = sum(1 for p in predictions if p[8] == 'Low')
    medium = sum(1 for p in predictions if p[8] == 'Medium')
    high = sum(1 for p in predictions if p[8] == 'High')

    return render_template('dashboard.html', 
                         predictions=predictions,
                         low=low, medium=medium, high=high,
                         total=len(predictions))


@app.route('/history')
@patient_required
def history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    user_id = session.get('user_id')
    c.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC', (user_id,))
    predictions = c.fetchall()
    conn.close()
    
    return render_template('history.html', predictions=predictions)


@app.route('/doctor')
@doctor_required
def doctor_dashboard():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT predictions.*, users.name 
        FROM predictions 
        LEFT JOIN users ON predictions.user_id = users.id
        ORDER BY predictions.id DESC
    ''')
    predictions = c.fetchall()
    conn.close()
    
    low = sum(1 for p in predictions if p[8] == 'Low')
    medium = sum(1 for p in predictions if p[8] == 'Medium')
    high = sum(1 for p in predictions if p[8] == 'High')
    
    return render_template('doctor.html',
                         predictions=predictions,
                         low=low, medium=medium, high=high,
                         total=len(predictions))


@app.route('/admin')
@admin_required
def admin_dashboard():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('SELECT * FROM users ORDER BY id DESC')
    users = c.fetchall()
    
    c.execute('SELECT COUNT(*) FROM predictions')
    total_predictions = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM users WHERE role = "doctor"')
    total_doctors = c.fetchone()[0]
    
    # Feedback fetch karo with user name
    c.execute('''
        SELECT feedback.*, users.name 
        FROM feedback 
        LEFT JOIN users ON feedback.user_id = users.id
        ORDER BY feedback.id DESC
    ''')
    feedbacks = c.fetchall()
    
    conn.close()
    
    return render_template('admin.html',
                         users=users,
                         total_users=len(users),
                         total_predictions=total_predictions,
                         total_doctors=total_doctors,
                         feedbacks=feedbacks)



@app.route('/report/<int:pred_id>')
@patient_required
def report(pred_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM predictions WHERE id = ?', (pred_id,))
    p = c.fetchone()
    conn.close()

    # Check 1: prediction exists at all
    if p is None:
        flash('Report not found.', 'error')
        return redirect(url_for('history'))

    # Check 2: prediction belongs to the logged-in patient
    # p[1] is the user_id column in the predictions table
    if p[1] != session.get('user_id'):
        flash('Access denied — this report does not belong to your account.', 'error')
        return redirect(url_for('history'))

    # Ownership confirmed — safe to generate PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 15, 'Diabetes Risk Prediction Report', ln=True, align='C')
    pdf.ln(5)
    
    # Date
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f'Date: {p[10]}', ln=True, align='C')
    pdf.ln(10)
    
    # Patient Info
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Patient Information', ln=True)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 8, f'Pregnancies: {p[2]}', ln=True)
    pdf.cell(0, 8, f'Glucose Level: {p[3]}', ln=True)
    pdf.cell(0, 8, f'Blood Pressure: {p[4]}', ln=True)
    pdf.cell(0, 8, f'BMI: {p[5]}', ln=True)
    pdf.cell(0, 8, f'Diabetes Pedigree: {p[6]}', ln=True)
    pdf.cell(0, 8, f'Age: {p[7]}', ln=True)

    pdf.ln(10)
    
    # Result
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Prediction Result', ln=True)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 8, f'Risk Level: {p[8]}', ln=True)
    pdf.cell(0, 8, f'Result: {p[9]}', ln=True)
    pdf.ln(10)
    
    # Recommendations
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Recommendations', ln=True)
    pdf.set_font('Helvetica', '', 12)
    
    if p[8] == 'Low':
        recs = ['Maintain a healthy diet', 'Stay physically active', 
                'Regular health checkups', 'Monitor glucose levels annually']
    elif p[8] == 'Medium':
        recs = ['Monitor glucose levels regularly', 'Reduce sugar and carb intake',
                'Consult a doctor soon', 'Exercise at least 30 mins daily']
    else:
        recs = ['Consult a doctor immediately', 'Strictly reduce sugar intake',
                'Monitor blood pressure daily', 'Follow prescribed medication']
    
    for rec in recs:
        pdf.cell(0, 8, f'- {rec}', ln=True)
    
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.cell(0, 8, 'Disclaimer: This report is for educational purposes only.', ln=True)

    from flask import Response
    import io

    response = pdf.output()
    return Response(
        bytes(response),
        mimetype='application/pdf',
        headers={
            'Content-Disposition': f'attachment; filename=diabetes_report_{pred_id}.pdf',
            'Content-Type': 'application/pdf'
        }
    )


@app.route('/add-doctor', methods=['POST'])
@admin_required
def add_doctor():
    name = request.form['name']
    email = request.form['email']
    password = generate_password_hash(request.form['password'])
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO users (name, email, password, role, date)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, email, password, 'doctor', datetime.now().strftime("%Y-%m-%d %H:%M")))
        conn.commit()
        flash('Doctor account created successfully!', 'success')
    except:
        flash('Email already exists.', 'error')
    conn.close()
    return redirect(url_for('admin_dashboard'))
    

@app.route('/education')
@patient_required
def education():
    return render_template('education.html')


@app.route('/feedback', methods=['GET', 'POST'])
@patient_required
def feedback():
    if request.method == 'POST':
        rating = request.form['rating']
        comment = request.form['comment']
        user_id = session.get('user_id')
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO feedback (user_id, rating, comment, date)
            VALUES (?, ?, ?, ?)
        ''', (user_id, rating, comment, datetime.now().strftime("%Y-%m-%d %H:%M")))
        conn.commit()
        conn.close()
        
        flash('Thank you for your feedback!', 'success')
        return redirect(url_for('home'))
    
    return render_template('feedback.html')



if __name__ == '__main__':
    app.run(debug=True)