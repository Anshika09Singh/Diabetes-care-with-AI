import os
from dotenv import load_dotenv
load_dotenv()  # load .env variables here
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
import os

# Google GenAI imports
from google import genai
from google.genai import types
from flask_cors import CORS

# Setup logging to stdout for clear debug info
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

# --- Load Machine Learning Model and Scaler ---
try:
    model = pickle.load(open('diabetes_model.pkl', 'rb'))
    logging.info("Diabetes Prediction Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading diabetes model: {e}")
    model = None

try:
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    logging.info("Scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading scaler: {e}")
    scaler = None

try:
    df = pd.read_csv('diabetes.csv')
    logging.info("Dataset loaded successfully.")
except Exception as e:
    logging.error(f"Error loading dataset: {e}")
    df = None


# --- Gemini Chatbot Integration ---
def get_gemini_response(user_message):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logging.error("GEMINI_API_KEY environment variable not set.")
            return "Error: Gemini API Key is not configured."

        client = genai.Client(api_key=api_key)
        model_name = "gemini-1.5-flash"

        # Initial user prompt for diabetes expert chatbot
        initial_content_user = types.Content(
            role="user",
            parts=[types.Part.from_text(text="""Extended Diabetes Expert Prompt:

You are a highly knowledgeable and empathetic medical assistant specializing in diabetes care... [trimmed for brevity, keep full prompt in your code]
""")]
        )
        # Initial model canned response
        initial_content_model = types.Content(
            role="model",
            parts=[types.Part.from_text(text="""Hello! I'm here to help you understand diabetes... [trimmed for brevity]
""")]
        )

        user_current_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )

        contents = [initial_content_user, initial_content_model, user_current_message]

        generate_content_config = types.GenerateContentConfig(response_mime_type="text/plain")

        full_response = ""
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        ):
            full_response += chunk.text
        return full_response

    except Exception as e:
        logging.error(f"Error getting Gemini response: {e}")
        return "Sorry, I'm having trouble connecting right now. Please try again later."


# --- Flask Routes ---
@app.route('/')
def root():
    return render_template('home.html')

@app.route('/index')
def home():
    logging.debug("Rendering index.html")
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        expected_features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

        features_dict = {}
        for feature_name in expected_features:
            value = request.form.get(feature_name)
            if value is None or value.strip() == "":
                logging.error(f"Missing feature: {feature_name}")
                return render_template('index.html', prediction_text=f"Error: Missing input for {feature_name}.")
            features_dict[feature_name] = float(value)

        features_list = [features_dict[f] for f in expected_features]
        logging.debug(f"Received features: {features_list}")

        if scaler is None or model is None:
            logging.error("Prediction model or scaler not loaded.")
            return render_template('index.html', prediction_text="Prediction model or scaler not loaded. Cannot perform prediction.")

        final_input = scaler.transform(np.array(features_list).reshape(1, -1))
        prediction = model.predict(final_input)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        logging.debug(f"Prediction result: {result}")

        return render_template('index.html', prediction_text=f"Prediction: {result}")

    except ValueError as ve:
        logging.error(f"Invalid input type: {ve}")
        return render_template('index.html', prediction_text="Invalid input. Please ensure all fields are numeric.")
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return render_template('index.html', prediction_text="An unexpected error occurred during prediction. Please try again.")


@app.route('/explore')
def explore():
    if df is None:
        logging.error("Dataset not loaded, cannot explore.")
        return "Dataset not loaded. Cannot generate plots.", 500

    try:
        # Correlation Heatmap
        corr = df.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of Diabetes Dataset Features', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        img_heatmap = io.BytesIO()
        plt.savefig(img_heatmap, format='png', bbox_inches='tight')
        plt.close()
        img_heatmap.seek(0)
        heatmap_url = base64.b64encode(img_heatmap.getvalue()).decode()
        logging.debug("Correlation heatmap generated.")

        # Glucose Histogram using Plotly Express
        fig_hist_glucose = px.histogram(df, x="Glucose", nbins=20, title="Distribution of Glucose Levels")
        fig_hist_glucose.update_layout(xaxis_title="Glucose Level", yaxis_title="Count")
        hist_glucose_html = fig_hist_glucose.to_html(full_html=False, include_plotlyjs='cdn')
        logging.debug("Glucose Histogram generated.")

        # BMI Histogram
        fig_hist_bmi = px.histogram(df, x="BMI", nbins=20, title="Distribution of BMI")
        fig_hist_bmi.update_layout(xaxis_title="BMI", yaxis_title="Count")
        hist_bmi_html = fig_hist_bmi.to_html(full_html=False, include_plotlyjs='cdn')
        logging.debug("BMI Histogram generated.")

        return render_template('explore.html',
                               heatmap_url=heatmap_url,
                               hist_glucose_html=hist_glucose_html,
                               hist_bmi_html=hist_bmi_html)
    except Exception as e:
        logging.error(f"Error in /explore route: {e}")
        return "Error generating plots. Check logs.", 500


@app.route('/chatbot')
def chatbot_page():
    logging.debug("Rendering chatbot.html")
    return render_template('chatbot.html')

@app.route('/life')
def life():
    logging.debug("Rendering life.html")
    return render_template('life.html')


@app.route('/generate', methods=['POST'])
def chat_gemini():
    data = request.get_json()
    logging.debug(f"Received JSON data: {data}")  # Add this line for debugging

    if not data:
        logging.warning("No JSON received in /generate request.")
        return jsonify({'reply': "Please send JSON data."}), 400

    user_message = data.get('message')
    if not user_message:
        logging.warning("No 'message' field in JSON data.")
        return jsonify({'reply': "Please provide a message."}), 400

    logging.debug(f"Received user message for Gemini: {user_message}")
    bot_response = get_gemini_response(user_message)
    logging.debug(f"Gemini bot response (first 100 chars): {bot_response[:100]}")

    return jsonify({'reply': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
