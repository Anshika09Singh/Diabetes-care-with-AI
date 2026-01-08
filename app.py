import os

from dotenv import load_dotenv

load_dotenv()

import base64
import io
import logging
import pickle
import re
import sys
import threading
from datetime import datetime, timezone

# ✅ Correct Gemini import
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from flask import Flask, jsonify, render_template, request, make_response
from flask_cors import CORS
from flask_mail import Mail, Message
from flask_babel import Babel, gettext as _

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
CORS(app)

# --- i18n Configuration ---
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Español',
    'hi': 'हिन्दी',
    'fr': 'Français',
    'zh': '中文'
}
DEFAULT_LANGUAGE = 'en'

app.config['BABEL_DEFAULT_LOCALE'] = DEFAULT_LANGUAGE
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'

babel = Babel(app)


def get_locale():
    # Check for user's language preference in cookie
    user_language = request.cookies.get('language')
    if user_language and user_language in SUPPORTED_LANGUAGES:
        return user_language
    
    # Fall back to browser's preferred language
    best_match = request.accept_languages.best_match(SUPPORTED_LANGUAGES.keys())
    if best_match:
        return best_match
    
    # Default to English
    return DEFAULT_LANGUAGE


babel.init_app(app, locale_selector=get_locale)


@app.context_processor
def inject_i18n_context():
    """Inject language context into all templates."""
    return {
        'languages': SUPPORTED_LANGUAGES,
        'current_language': get_locale(),
        'current_year': datetime.now().year,
    }


@app.route('/api/set-language', methods=['POST'])
def set_language():
    """Set user's preferred language via cookie."""
    data = request.get_json()
    
    if not data or 'language' not in data:
        return jsonify({'error': 'Language code is required'}), 400
    
    language = data['language']
    
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({
            'error': 'Unsupported language',
            'supported': list(SUPPORTED_LANGUAGES.keys())
        }), 400
    
    response = make_response(jsonify({
        'success': True,
        'language': language,
        'language_name': SUPPORTED_LANGUAGES[language]
    }))
    
    # Set cookie for 1 year
    response.set_cookie('language', language, max_age=365*24*60*60, samesite='Lax')
    
    return response


# Flask-Mail Configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'true').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', 'noreply@diabetescare.com')

mail = Mail(app)

# In-memory store for forum posts
posts = []

# Available topics for posts
FORUM_TOPICS = ['General', 'Diet', 'Exercise', 'Medication', 'Lifestyle', 'Support']

# In-memory store for users and notifications
users = {}  # {user_id: {email, username, preferences, subscribed_posts}}
notifications = []  # [{id, user_id, type, message, post_id, read, timestamp}]

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
        model = genai.GenerativeModel("gemini-2.5-flash")

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
            return render_template('index.html', prediction_text=_("Model not available."))

        final_input = scaler.transform([features])
        prediction = model.predict(final_input)
        result = _("Diabetic") if prediction[0] == 1 else _("Not Diabetic")

        return render_template('index.html', prediction_text=_("Prediction: %(result)s", result=result))
    except Exception as e:
        logging.error(f"Predict error: {e}")
        return render_template('index.html', prediction_text=_("Error during prediction."))


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


@app.route('/api/lifestyle-plan', methods=['POST'])
def generate_lifestyle_plan():
    """Generate a personalized lifestyle plan based on user inputs."""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    age = data.get('age')
    bmi = data.get('bmi')
    activity = data.get('activity')
    
    # Validate inputs
    if not all([age, bmi, activity]):
        return jsonify({'error': 'Age, BMI, and activity level are required'}), 400
    
    try:
        age = int(age)
        bmi = float(bmi)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid age or BMI value'}), 400
    
    if age < 1 or age > 120:
        return jsonify({'error': 'Age must be between 1 and 120'}), 400
    if bmi < 10 or bmi > 50:
        return jsonify({'error': 'BMI must be between 10 and 50'}), 400
    if activity not in ['low', 'moderate', 'high']:
        return jsonify({'error': 'Activity must be low, moderate, or high'}), 400
    
    # Determine BMI category
    if bmi < 18.5:
        bmi_category = 'underweight'
    elif bmi < 25:
        bmi_category = 'normal weight'
    elif bmi < 30:
        bmi_category = 'overweight'
    else:
        bmi_category = 'obese'
    
    # Determine age group
    if age < 30:
        age_group = 'young adult'
    elif age < 50:
        age_group = 'middle-aged adult'
    elif age < 65:
        age_group = 'older adult'
    else:
        age_group = 'senior'
    
    # Try to get AI-generated plan
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            ai_model = genai.GenerativeModel("gemini-2.5-flash")
            
            prompt = f"""Generate a personalized lifestyle plan for diabetes prevention/management with these details:
- Age: {age} years ({age_group})
- BMI: {bmi} ({bmi_category})
- Physical Activity Level: {activity}

Provide specific, actionable recommendations in these categories:
1. Diet (3-4 specific tips)
2. Exercise (3-4 specific recommendations based on their activity level)
3. Stress Management (2-3 techniques)
4. Sleep & Recovery (2-3 tips)
5. Health Monitoring (2-3 suggestions)

Keep each tip concise (1-2 sentences). Focus on diabetes-relevant advice. Format as JSON with keys: diet, exercise, stress, sleep, monitoring (each containing an array of strings)."""

            response = ai_model.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON from response
            import json
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                plan = json.loads(json_match.group())
                return jsonify({
                    'success': True,
                    'plan': plan,
                    'profile': {
                        'age': age,
                        'age_group': age_group,
                        'bmi': bmi,
                        'bmi_category': bmi_category,
                        'activity': activity
                    },
                    'source': 'ai'
                })
    except Exception as e:
        logging.warning(f"AI plan generation failed, using fallback: {e}")
    
    # Fallback: Rule-based plan generation
    plan = generate_fallback_plan(age, bmi, bmi_category, activity, age_group)
    
    return jsonify({
        'success': True,
        'plan': plan,
        'profile': {
            'age': age,
            'age_group': age_group,
            'bmi': bmi,
            'bmi_category': bmi_category,
            'activity': activity
        },
        'source': 'rules'
    })


def generate_fallback_plan(age, bmi, bmi_category, activity, age_group):
    """Generate a rule-based lifestyle plan as fallback."""
    plan = {'diet': [], 'exercise': [], 'stress': [], 'sleep': [], 'monitoring': []}
    
    # Diet recommendations based on BMI
    if bmi_category == 'underweight':
        plan['diet'] = [
            'Increase caloric intake with nutrient-dense foods like nuts, avocados, and whole grains.',
            'Eat 5-6 smaller meals throughout the day to boost calorie consumption.',
            'Include protein-rich foods at every meal to support healthy weight gain.',
            'Consider healthy smoothies with fruits, yogurt, and nut butters.'
        ]
    elif bmi_category == 'normal weight':
        plan['diet'] = [
            'Maintain a balanced diet with plenty of vegetables, lean proteins, and whole grains.',
            'Practice portion control and mindful eating habits.',
            'Limit processed foods and added sugars to prevent blood sugar spikes.',
            'Stay hydrated with water and limit sugary beverages.'
        ]
    elif bmi_category == 'overweight':
        plan['diet'] = [
            'Focus on a moderate calorie deficit with nutrient-dense, low-glycemic foods.',
            'Fill half your plate with non-starchy vegetables at each meal.',
            'Choose lean proteins and limit saturated fats.',
            'Reduce refined carbohydrates and opt for whole grain alternatives.'
        ]
    else:  # obese
        plan['diet'] = [
            'Work with a dietitian to create a sustainable calorie-controlled meal plan.',
            'Prioritize high-fiber foods to improve satiety and blood sugar control.',
            'Eliminate sugary drinks and limit fruit juices.',
            'Practice meal prepping to avoid unhealthy food choices.'
        ]
    
    # Exercise recommendations based on activity level
    if activity == 'low':
        plan['exercise'] = [
            'Start with 15-20 minute walks daily and gradually increase duration.',
            'Try chair exercises or gentle stretching if mobility is limited.',
            'Set reminders to stand and move every hour if you have a sedentary job.',
            'Consider swimming or water aerobics for low-impact cardio.'
        ]
    elif activity == 'moderate':
        plan['exercise'] = [
            'Aim for 150 minutes of moderate aerobic activity per week.',
            'Add 2-3 strength training sessions to build muscle and improve insulin sensitivity.',
            'Include flexibility exercises like yoga or stretching.',
            'Try interval training to boost cardiovascular health.'
        ]
    else:  # high
        plan['exercise'] = [
            'Maintain your excellent activity level with varied workouts.',
            'Ensure adequate rest days to prevent overtraining.',
            'Focus on proper nutrition timing around workouts.',
            'Consider working with a trainer to optimize your routine.'
        ]
    
    # Stress management based on age
    if age >= 50:
        plan['stress'] = [
            'Practice daily meditation or deep breathing exercises for 10-15 minutes.',
            'Join social groups or community activities to stay connected.',
            'Consider gentle yoga or tai chi for combined physical and mental benefits.'
        ]
    else:
        plan['stress'] = [
            'Identify stress triggers and develop healthy coping mechanisms.',
            'Schedule regular breaks and leisure activities.',
            'Practice mindfulness or use stress-management apps.'
        ]
    
    # Sleep recommendations
    if age >= 65:
        plan['sleep'] = [
            'Aim for 7-8 hours of sleep with a consistent bedtime routine.',
            'Limit daytime naps to 20-30 minutes if needed.',
            'Avoid screens and caffeine at least 2 hours before bed.'
        ]
    else:
        plan['sleep'] = [
            'Prioritize 7-9 hours of quality sleep each night.',
            'Create a dark, cool, and quiet sleep environment.',
            'Maintain consistent sleep and wake times, even on weekends.'
        ]
    
    # Health monitoring
    plan['monitoring'] = [
        'Check blood glucose levels as recommended by your healthcare provider.',
        'Monitor blood pressure regularly, especially if overweight.',
        'Schedule regular check-ups and diabetes screenings.'
    ]
    
    return plan


@app.route('/generate', methods=['POST'])
def chat_gemini():
    data = request.get_json()
    if not data or not data.get('message'):
        return jsonify({'reply': "Please provide a message."}), 400

    user_message = data['message']
    bot_response = get_gemini_response(user_message)
    return jsonify({'reply': bot_response})


# --- Forum Backend ---

def filter_posts(posts_list, search=None, start_date=None, end_date=None, topic=None):
    """
    Filter posts by search term, date range, and topic.

    Args:
        posts_list: List of post dictionaries
        search: Optional search string (case-insensitive)
        start_date: Optional datetime for minimum date
        end_date: Optional datetime for maximum date
        topic: Optional topic string

    Returns:
        List of filtered posts
    """
    filtered = posts_list.copy()

    # Apply search filter (case-insensitive)
    if search and search.strip():
        search_lower = search.lower()
        filtered = [p for p in filtered if search_lower in p.get('content', '').lower()]

    # Apply topic filter
    if topic and topic.strip():
        filtered = [p for p in filtered if p.get('topic') == topic]

    # Apply start_date filter
    if start_date:
        filtered = [p for p in filtered if parse_post_timestamp(p.get('timestamp')) >= start_date]

    # Apply end_date filter
    if end_date:
        filtered = [p for p in filtered if parse_post_timestamp(p.get('timestamp')) <= end_date]

    return filtered


def parse_post_timestamp(timestamp_str):
    """
    Parse ISO timestamp string to datetime object.
    Handles timestamps with or without 'Z' suffix.
    Returns UTC datetime objects for consistent timezone handling.
    """
    if not timestamp_str:
        return datetime.min.replace(tzinfo=timezone.utc)

    # Remove 'Z' suffix if present and parse
    ts = timestamp_str.rstrip('Z')
    try:
        dt = datetime.fromisoformat(ts)
        # If the datetime is naive, assume it's UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def paginate_posts(posts_list, page=1, per_page=10):
    """
    Paginate a list of posts.
    
    Args:
        posts_list: List of posts to paginate
        page: Current page number (1-indexed)
        per_page: Number of posts per page
    
    Returns:
        dict: {
            'posts': List of posts for current page,
            'total': Total number of posts,
            'page': Current page number,
            'per_page': Posts per page,
            'total_pages': Total number of pages
        }
    """
    import math
    
    total = len(posts_list)
    total_pages = math.ceil(total / per_page) if total > 0 else 1
    
    # Ensure page is within valid range
    page = max(1, min(page, total_pages))
    
    # Calculate slice indices
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    return {
        'posts': posts_list[start_idx:end_idx],
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': total_pages
    }


@app.route('/forum')
def forum():
    return render_template('forum.html')


@app.route('/api/posts', methods=['GET', 'POST'])
def posts_api():
    if request.method == 'GET':
        # Parse query parameters
        search = request.args.get('search', '').strip() or None
        start_date_str = request.args.get('start_date', '').strip()
        end_date_str = request.args.get('end_date', '').strip()
        topic = request.args.get('topic', '').strip() or None

        try:
            page = int(request.args.get('page', 1))
            if page < 1:
                return jsonify({"error": "Page must be a positive integer"}), 400
        except ValueError:
            return jsonify({"error": "Page must be a positive integer"}), 400

        try:
            per_page = int(request.args.get('per_page', 10))
            if per_page < 1 or per_page > 50:
                return jsonify({"error": "per_page must be between 1 and 50"}), 400
        except ValueError:
            return jsonify({"error": "per_page must be between 1 and 50"}), 400

        # Parse dates (make timezone-aware for consistent comparison)
        start_date = None
        end_date = None

        if start_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str)
                # Make timezone-aware (UTC) for consistent comparison
                if start_date.tzinfo is None:
                    start_date = start_date.replace(tzinfo=timezone.utc)
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str)
                # Set end_date to end of day for inclusive filtering
                end_date = end_date.replace(hour=23, minute=59, second=59)
                # Make timezone-aware (UTC) for consistent comparison
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=timezone.utc)
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

        # Validate date range
        if start_date and end_date and start_date > end_date:
            return jsonify({"error": "start_date must be before or equal to end_date"}), 400

        # Sort posts by timestamp (newest first)
        sorted_posts = sorted(posts, key=lambda x: x['timestamp'], reverse=True)

        # Apply filters
        filtered_posts = filter_posts(sorted_posts, search=search, start_date=start_date, end_date=end_date, topic=topic)

        # Paginate results
        result = paginate_posts(filtered_posts, page=page, per_page=per_page)


        return jsonify({
            'posts': result['posts'],
            'pagination': {
                'total': result['total'],
                'page': result['page'],
                'per_page': result['per_page'],
                'total_pages': result['total_pages']
            }
        })
    
    elif request.method == 'POST':
        data = request.json
        content = data.get('content', '').strip()
        author_id = data.get('author_id', 'anonymous')
        parent_id = data.get('parent_id')  # For replies
        topic = data.get('topic', '').strip() or None

        if not content:
            return jsonify({"error": "Content is required"}), 400

        if topic and topic not in FORUM_TOPICS:
            return jsonify({"error": f"Invalid topic. Available topics: {FORUM_TOPICS}"}), 400

        post = {
            'id': len(posts) + 1,
            'content': content,
            'author_id': author_id,
            'parent_id': parent_id,
            'topic': topic,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        posts.append(post)

        # Process notifications asynchronously
        threading.Thread(target=process_post_notifications, args=(post,)).start()

        return jsonify(post), 201


# --- Notification System ---

def extract_mentions(content):
    """Extract @username mentions from post content."""
    return re.findall(r'@(\w+)', content)


def send_email_notification(to_email, subject, body):
    """Send email notification asynchronously."""
    try:
        if not app.config['MAIL_USERNAME']:
            logging.warning("Email not configured, skipping notification")
            return False
        
        with app.app_context():
            msg = Message(subject=subject, recipients=[to_email], body=body)
            mail.send(msg)
            logging.info(f"Email sent to {to_email}")
            return True
    except Exception as e:
        logging.error(f"Failed to send email: {e}")
        return False


def create_notification(user_id, notif_type, message, post_id=None):
    """Create an in-app notification."""
    notification = {
        'id': len(notifications) + 1,
        'user_id': user_id,
        'type': notif_type,
        'message': message,
        'post_id': post_id,
        'read': False,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    notifications.append(notification)
    return notification


def process_post_notifications(post):
    """Process notifications for a new post (replies, mentions)."""
    content = post.get('content', '')
    author_id = post.get('author_id', 'anonymous')
    post_id = post.get('id')
    parent_id = post.get('parent_id')
    
    # Handle reply notifications
    if parent_id:
        parent_post = next((p for p in posts if p['id'] == parent_id), None)
        if parent_post and parent_post.get('author_id') != author_id:
            original_author = parent_post.get('author_id')
            if original_author in users:
                user = users[original_author]
                # Create in-app notification
                create_notification(
                    original_author, 
                    'reply', 
                    f"Someone replied to your post",
                    post_id
                )
                # Send email if enabled
                if user.get('preferences', {}).get('email_replies', True):
                    send_email_notification(
                        user['email'],
                        "New reply to your post - Diabetes Care Forum",
                        f"Someone replied to your post:\n\n{content[:200]}...\n\nVisit the forum to see the full reply."
                    )
    
    # Handle mention notifications
    mentions = extract_mentions(content)
    logging.info(f"Processing mentions: {mentions}, registered users: {list(users.keys())}")
    
    # Get author's username for the notification message
    author_username = users.get(author_id, {}).get('username', 'Someone')
    
    for username in mentions:
        # Find user by username (case-insensitive)
        mentioned_user = next(
            (uid for uid, u in users.items() if u.get('username', '').lower() == username.lower()),
            None
        )
        logging.info(f"Looking for @{username}, found user_id: {mentioned_user}")
        if mentioned_user and mentioned_user != author_id:
            user = users[mentioned_user]
            create_notification(
                mentioned_user,
                'mention',
                f"@{author_username} mentioned you in a post",
                post_id
            )
            logging.info(f"Created mention notification for {mentioned_user}")
            if user.get('preferences', {}).get('email_mentions', True):
                send_email_notification(
                    user['email'],
                    "You were mentioned - Diabetes Care Forum",
                    f"You were mentioned in a post:\n\n{content[:200]}...\n\nVisit the forum to see the full post."
                )
    
    # Notify subscribers of the thread
    if parent_id:
        for uid, user in users.items():
            if parent_id in user.get('subscribed_posts', []) and uid != author_id:
                create_notification(
                    uid,
                    'subscription',
                    f"New activity in a thread you're following",
                    post_id
                )
                if user.get('preferences', {}).get('email_subscriptions', True):
                    send_email_notification(
                        user['email'],
                        "New activity in subscribed thread - Diabetes Care Forum",
                        f"There's new activity in a thread you're following:\n\n{content[:200]}..."
                    )


# --- User & Notification API Endpoints ---

@app.route('/api/users', methods=['GET', 'POST'])
def users_api():
    """Register or update a user for notifications, or list all users."""
    if request.method == 'GET':
        # Debug endpoint to see registered users
        return jsonify({"users": {uid: {"username": u.get("username"), "email": u.get("email")} for uid, u in users.items()}})
    
    data = request.json
    user_id = data.get('user_id')
    email = data.get('email', '').strip()
    username = data.get('username', '').strip()
    
    if not user_id or not email:
        return jsonify({"error": "user_id and email are required"}), 400
    
    if user_id in users:
        # Update existing user
        users[user_id]['email'] = email
        if username:
            users[user_id]['username'] = username
    else:
        # Create new user
        users[user_id] = {
            'email': email,
            'username': username or user_id,
            'preferences': {
                'email_replies': True,
                'email_mentions': True,
                'email_subscriptions': True
            },
            'subscribed_posts': []
        }
    
    logging.info(f"User registered/updated: {user_id} -> {users[user_id]}")
    return jsonify({"message": "User registered", "user": users[user_id]}), 200


@app.route('/api/users/<user_id>/preferences', methods=['GET', 'PUT'])
def user_preferences(user_id):
    """Get or update user notification preferences."""
    if user_id not in users:
        return jsonify({"error": "User not found"}), 404
    
    if request.method == 'GET':
        return jsonify(users[user_id].get('preferences', {}))
    
    elif request.method == 'PUT':
        data = request.json
        prefs = users[user_id].get('preferences', {})
        
        if 'email_replies' in data:
            prefs['email_replies'] = bool(data['email_replies'])
        if 'email_mentions' in data:
            prefs['email_mentions'] = bool(data['email_mentions'])
        if 'email_subscriptions' in data:
            prefs['email_subscriptions'] = bool(data['email_subscriptions'])
        
        users[user_id]['preferences'] = prefs
        return jsonify({"message": "Preferences updated", "preferences": prefs})


@app.route('/api/users/<user_id>/subscribe/<int:post_id>', methods=['POST', 'DELETE'])
def subscribe_post(user_id, post_id):
    """Subscribe or unsubscribe from a post/thread."""
    if user_id not in users:
        return jsonify({"error": "User not found"}), 404
    
    subscribed = users[user_id].setdefault('subscribed_posts', [])
    
    if request.method == 'POST':
        if post_id not in subscribed:
            subscribed.append(post_id)
        return jsonify({"message": "Subscribed", "subscribed_posts": subscribed})
    
    elif request.method == 'DELETE':
        if post_id in subscribed:
            subscribed.remove(post_id)
        return jsonify({"message": "Unsubscribed", "subscribed_posts": subscribed})


@app.route('/api/notifications/<user_id>', methods=['GET'])
def get_notifications(user_id):
    """Get notifications for a user."""
    # Allow fetching even if user not fully registered yet
    user_notifications = [n for n in notifications if n['user_id'] == user_id]
    
    # Sort by timestamp descending
    user_notifications.sort(key=lambda x: x['timestamp'], reverse=True)
    
    unread_count = sum(1 for n in user_notifications if not n['read'])
    
    return jsonify({
        "notifications": user_notifications[:50],
        "unread_count": unread_count
    })


@app.route('/api/notifications/<user_id>/read', methods=['POST'])
def mark_notifications_read(user_id):
    """Mark notifications as read and optionally delete them."""
    data = request.json or {}
    notification_ids = data.get('notification_ids', [])
    mark_all = data.get('mark_all', False)
    delete_read = data.get('delete', True)  # Delete by default
    
    global notifications
    count = 0
    
    if delete_read:
        # Remove notifications instead of just marking as read
        if mark_all:
            before_count = len([n for n in notifications if n['user_id'] == user_id])
            notifications = [n for n in notifications if n['user_id'] != user_id]
            count = before_count
        else:
            before_count = len(notifications)
            notifications = [n for n in notifications if not (n['user_id'] == user_id and n['id'] in notification_ids)]
            count = before_count - len(notifications)
    else:
        # Just mark as read
        for n in notifications:
            if n['user_id'] == user_id:
                if mark_all or n['id'] in notification_ids:
                    if not n['read']:
                        n['read'] = True
                        count += 1
    
    return jsonify({"message": f"Processed {count} notifications"})


# --- Run App ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
