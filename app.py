import os
from flask import Flask, render_template, session, redirect, url_for, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'saas-secret-key-001')

# Database Setup - Railway Volume পাথ
DB_PATH = os.path.join('/data', 'users.db') if os.path.exists('/data') else 'users.db'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True)
    tokens = db.Column(db.Text)
    gemini_key = db.Column(db.String(200))
    drive_id = db.Column(db.String(200))

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    user = User.query.filter_by(email=session.get('user_email')).first() if 'user_email' in session else None
    return render_template('index.html', user=user)

# আপনার অন্য সব রুট (login, callback) এখানে থাকবে...

if __name__ == '__main__':
    # Railway-এর জন্য পোর্ট সেট করা (Healthcheck ঠিক করতে এটি জরুরি)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
