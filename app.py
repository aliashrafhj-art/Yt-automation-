import os
from flask import Flask, render_template, session, redirect, url_for, request

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'final-prod-key-786')

# Railway Healthcheck Route
@app.route('/check')
def health_check():
    return "OK", 200

@app.route('/')
def index():
    return "<h1>Server is Running!</h1><p>Go to your dashboard or login.</p>"

# অন্য সব রুট এখানে থাকবে...

if __name__ == '__main__':
    # Railway-এর জন্য পোর্ট এবং হোস্ট ফিক্সিং
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
