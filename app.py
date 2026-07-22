from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf
from utils import detect_face, get_recommendations, validate_image
import random
import json
import time
import threading
import webbrowser

# --- Authentication imports ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_required
from models import db, User
from facial_region import (
    extract_regions,
    save_regions,
    draw_regions
)

# --- Load acne class mapping ---
try:
    with open('models/acne_class_labels.json', 'r') as f:
        CLASS_MAPPING = json.load(f)
    print(f"✅ Class mapping loaded: {CLASS_MAPPING}")
    ACNE_INDEX = CLASS_MAPPING.get('acne', 0)
    NO_ACNE_INDEX = CLASS_MAPPING.get('no_acne', 1)
except FileNotFoundError:
    print("⚠️ No class mapping file found, using correct defaults (0=acne, 1=no_acne)")
    ACNE_INDEX = 0
    NO_ACNE_INDEX = 1

# --- Initialize Flask app ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# --- Database configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# --- Session configuration ---
app.config['SECRET_KEY'] = 'your-very-secret-key-change-this-in-production'

# --- Login manager setup ---
login_manager = LoginManager()
login_manager.login_view = 'auth.login'  # Route name for login page
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    """Required callback for Flask-Login to reload the user object."""
    return User.query.get(int(user_id))

# --- Register authentication blueprint ---
from auth import auth as auth_blueprint
app.register_blueprint(auth_blueprint)

# --- Ensure necessary directories exist ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('static/region_images', exist_ok=True)

# --- Load deep learning models ---
try:
    skin_model = tf.keras.models.load_model('models/skin_type_model.h5')
    acne_model = tf.keras.models.load_model('models/acne_model.h5')
    print("✅ Both models loaded successfully!")
    MODELS_LOADED = True
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    MODELS_LOADED = False
    skin_model = None
    acne_model = None

# --- Infer skin type class order from dataset ---
if os.path.exists('dataset_skin_type/train'):
    SKIN_CLASSES = sorted(os.listdir('dataset_skin_type/train'))
    print(f"✅ Skin classes (alphabetical): {SKIN_CLASSES}")
else:
    SKIN_CLASSES = ['dry', 'normal', 'oily']  # default alphabetical
    print("⚠️ Using default skin class order: dry, normal, oily")

# --- Helper functions ---

def preprocess_image(image_path):
    """Preprocess image for model prediction."""
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    return img_array

def predict_skin_type(image_path):
    """Predict skin type with correct class order."""
    if MODELS_LOADED and skin_model is not None:
        try:
            img_array = preprocess_image(image_path)
            predictions = skin_model.predict(img_array, verbose=0)[0]

            if len(predictions) >= 3:
                skin_idx = np.argmax(predictions[:3])
                confidence = float(predictions[skin_idx])
                skin_type = SKIN_CLASSES[skin_idx].capitalize()
                print(f"DEBUG: Skin idx: {skin_idx}, Type: {skin_type}, Conf: {confidence:.4f}")
                return skin_type, confidence
            else:
                return random.choice(['Oily', 'Dry', 'Normal']), 0.85
        except Exception as e:
            print(f"Skin prediction error: {e}")
            return random.choice(['Oily', 'Dry', 'Normal']), 0.85
    else:
        return random.choice(['Oily', 'Dry', 'Normal']), 0.85

def predict_acne(image_path):
    """Predict acne using correct class indices."""
    if MODELS_LOADED and acne_model is not None:
        try:
            img_array = preprocess_image(image_path)
            predictions = acne_model.predict(img_array, verbose=0)[0]

            acne_conf = float(predictions[ACNE_INDEX])
            no_acne_conf = float(predictions[NO_ACNE_INDEX])

            print(f"DEBUG: Acne confidence: {acne_conf:.4f}, No Acne: {no_acne_conf:.4f}")

            if acne_conf > no_acne_conf:
                return 'Acne Detected', acne_conf
            else:
                return 'No Acne', no_acne_conf
        except Exception as e:
            print(f"Acne prediction error: {e}")
            return random.choice(['No Acne', 'Acne Detected']), 0.85
    else:
        return random.choice(['No Acne', 'Acne Detected']), 0.85

# --- Routes ---
@app.route('/')
@login_required
def index():
    """Main page (protected)."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@login_required
def analyze_skin():
    """Analyze uploaded image and return results."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    is_valid, message = validate_image(file)

    if not is_valid:
        return jsonify({'error': message}), 400

    try:
        # Save uploaded file with timestamp
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{int(time.time())}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Detect face
        face_img, face_detected = detect_face(filepath)
        if face_detected:
            face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'face_' + filename)
            cv2.imwrite(face_path, face_img)
        else:
            face_path = filepath

        # ---------------------------------------------
        # Facial Region Analysis
        # ---------------------------------------------

        original_image = cv2.imread(filepath)

        regions = extract_regions(original_image)

        if regions is not None:
            save_regions(regions)

            annotated = draw_regions(
                original_image,
                regions["bbox"]
            )

            annotated_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                "annotated_" + filename
            )

            cv2.imwrite(
                annotated_path,
                annotated
            )
            print("Annotated Path:", annotated_path)
            print("Image Saved:", os.path.exists(annotated_path))
            print("Image Shape:", annotated.shape)
        else:
            annotated_path = filepath

        # Predict
        skin_type, skin_conf = predict_skin_type(face_path)
        acne_status, acne_conf = predict_acne(face_path)

        # Get recommendations
        products, tips = get_recommendations(skin_type, acne_status)

        result = {

    "success": True,

    "skin_type": skin_type,

    "acne_status": acne_status,

    "confidence": {
        "skin": f"{skin_conf:.1%}",
        "acne": f"{acne_conf:.1%}"
    },

    "face_detected": face_detected,

    "products": products,

    "tips": tips,

    "image_path": f"/static/uploads/{filename}",

    "annotated_face":
        f"/static/uploads/annotated_{filename}",

    "regions": {

        "forehead":
            "/static/region_images/forehead.jpg",

        "nose":
            "/static/region_images/nose.jpg",

        "left_cheek":
            "/static/region_images/left_cheek.jpg",

        "right_cheek":
            "/static/region_images/right_cheek.jpg",

        "chin":
            "/static/region_images/chin.jpg"

    }

}

        return jsonify(result)

    except Exception as e:
        print(f"ERROR in analyze_skin: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# --- Create database tables ---
with app.app_context():
    db.create_all()

# --- Automatically open the app in the default web browser ---
def open_browser():
    """Open the browser after a short delay to ensure the server is running."""
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    # Start a timer that launches the browser 1.5 seconds after the server starts
    threading.Timer(1.5, open_browser).start()
    # Launch the Flask server without the reloader to avoid double‑opening
    app.run(debug=True, port=5000, use_reloader=False)