import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import logging
import sys
from datetime import datetime
from flask_cors import CORS

# ✅ Correct Gemini import
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# In-memory store for forum posts
posts = []

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')

# --- Load ML Model and Scaler ---
try:
    model = pickle.load(open('diabetes_model.pkl', 'rb'))
    logging.info("Diabetes model loaded successfully.")
except Exception as e:
    logging.error(f"Model load error: {e}")
    model = None

try:
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    logging.info("Scaler loaded successfully.")
except Exception as e:
    logging.error(f"Scaler load error: {e}")
    scaler = None

try:
    df = pd.read_csv('diabetes.csv')
    logging.info("Dataset loaded.")
except Exception as e:
    logging.error(f"CSV load error: {e}")
    df = None


# --- Gemini AI Chat Function ---
def get_gemini_response(user_message):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logging.error("GEMINI_API_KEY not set.")
            return "Error: Gemini API Key not found."

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        chat = model.start_chat(history=[
            {"role": "user", "parts": ["You're a helpful diabetes assistant."]}
        ])
        response = chat.send_message(user_message)
        return response.text
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "Sorry, Gemini service is unavailable right now."


# --- Flask Routes ---
@app.route('/')
def root():
    return render_template('home.html')


@app.route('/index')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        expected_features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        features = [float(request.form.get(f, 0)) for f in expected_features]

        if scaler is None or model is None:
            return render_template('index.html', prediction_text="Model not available.")

        final_input = scaler.transform([features])
        prediction = model.predict(final_input)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        return render_template('index.html', prediction_text=f"Prediction: {result}")
    except Exception as e:
        logging.error(f"Predict error: {e}")
        return render_template('index.html', prediction_text="Error during prediction.")


@app.route('/explore')
def explore():
    if df is None:
        return "Dataset not loaded", 500
    try:
        corr = df.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        heatmap_url = base64.b64encode(img.getvalue()).decode()

        fig_glucose = px.histogram(df, x="Glucose", nbins=20, title="Glucose Distribution")
        fig_bmi = px.histogram(df, x="BMI", nbins=20, title="BMI Distribution")

        return render_template('explore.html',
                               heatmap_url=heatmap_url,
                               hist_glucose_html=fig_glucose.to_html(full_html=False, include_plotlyjs='cdn'),
                               hist_bmi_html=fig_bmi.to_html(full_html=False, include_plotlyjs='cdn'))
    except Exception as e:
        logging.error(f"Explore error: {e}")
        return "Error generating plots", 500


@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')


@app.route('/life')
def life():
    return render_template('life.html')


@app.route('/generate', methods=['POST'])
def chat_gemini():
    data = request.get_json()
    if not data or not data.get('message'):
        return jsonify({'reply': "Please provide a message."}), 400

    user_message = data['message']
    bot_response = get_gemini_response(user_message)
    return jsonify({'reply': bot_response})


# --- Forum Backend ---
@app.route('/forum')
def forum():
    return render_template('forum.html')


@app.route('/api/posts', methods=['GET', 'POST'])
def posts_api():
    if request.method == 'GET':
        return jsonify(sorted(posts, key=lambda x: x['timestamp'], reverse=True))
    elif request.method == 'POST':
        data = request.json
        content = data.get('content', '').strip()
        if not content:
            return jsonify({"error": "Content is required"}), 400

        post = {
            'id': len(posts) + 1,
            'content': content,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        posts.append(post)
        return jsonify(post), 201


# --- Run App ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
