import sqlite3
import os
from werkzeug.security import generate_password_hash
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, 'diabetes.db')

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create users table if app.py has not been run yet
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL,
        date TEXT
    )
''')
conn.commit()

admin_email = "admin@gmail.com"

cursor.execute("SELECT id FROM users WHERE email = ?", (admin_email,))
existing_admin = cursor.fetchone()

if existing_admin:
    print("Admin already exists. No new admin was created.")
else:
    hashed_password = generate_password_hash("admin123")

    cursor.execute("""
    INSERT INTO users (name, email, password, role, date)
    VALUES (?, ?, ?, ?, ?)
    """, (
        "Admin",
        admin_email,
        hashed_password,
        "admin",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    print("Admin created successfully!")

conn.close()