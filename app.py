import os, json
from flask import Flask, render_template, session, redirect, url_for, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-key-123')

# Database Setup - রেলওয়ে ভলিউম পাথ
DB_DIR = '/data' if os.path.exists('/data') else '.'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(DB_DIR, "users.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
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
    user = None
    if 'user_email' in session:
        user = User.query.filter_by(email=session['user_email']).first()
    return render_template('index.html', user=user)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Healthcheck failure সমাধান করতে এই পোর্ট সেটিংস জরুরি
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
