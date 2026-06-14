import cv2
import numpy as np
import json
import os
from PIL import Image

def detect_face(image_path):
    """
    Detect face in image using OpenCV
    """
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return None, False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # Get the first face
        (x, y, w, h) = faces[0]
        # Crop face with some margin
        margin = 30
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2*margin)
        h = min(img.shape[0] - y, h + 2*margin)
        
        face_img = img[y:y+h, x:x+w]
        return face_img, True
    else:
        return img, False

def load_product_database():
    """
    Load product database from JSON file
    """
    try:
        with open('data/products.json', 'r') as f:
            data = json.load(f)
        return data
    except:
        # Return mock data if file doesn't exist
        return {
            "products": [],
            "skincare_tips": {
                "Oily": ["Wash face twice daily", "Use oil-free products"],
                "Dry": ["Use gentle cleanser", "Apply moisturizer regularly"],
                "Normal": ["Maintain regular routine", "Use sunscreen daily"]
            }
        }
def get_recommendations(skin_type, acne_status):
    """
    Get product recommendations based on skin type and acne status.
    Returns at least 3 and at most 5 products.
    """
    data = load_product_database()
    products = data.get('products', [])
    tips = data.get('skincare_tips', {})

    perfect_matches = []
    partial_matches = []
    fallback_matches = []

    # Categorize products
    for product in products:
        if skin_type in product.get('skin_type', []):
            if acne_status == 'Acne Detected':
                if 'Acne' in product.get('skin_concern', []):
                    perfect_matches.append(product)
                else:
                    partial_matches.append(product)
            else:
                if 'Acne' not in product.get('skin_concern', []):
                    perfect_matches.append(product)
                else:
                    partial_matches.append(product)
        else:
            # Fallback: products for all skin types or any product if needed
            if 'All' in product.get('skin_type', []) or not product.get('skin_type'):
                fallback_matches.append(product)

    # Build recommended list in priority order
    recommended = []
    
    # First, add perfect matches (up to 5)
    for p in perfect_matches:
        if len(recommended) >= 5:
            break
        recommended.append(p)
    
    # If still less than 5, add partial matches
    if len(recommended) < 5:
        for p in partial_matches:
            if len(recommended) >= 5:
                break
            if p not in recommended:
                recommended.append(p)
    
    # If still less than 3, add fallback matches (products for all skin types)
    if len(recommended) < 3:
        for p in fallback_matches:
            if len(recommended) >= 5:
                break
            if p not in recommended:
                recommended.append(p)
    
    # If still less than 3, add any remaining products (ignore skin type)
    if len(recommended) < 3:
        for p in products:
            if len(recommended) >= 5:
                break
            if p not in recommended:
                recommended.append(p)
    
    # Get skincare tips
    skin_tips = tips.get(skin_type, ["Maintain regular skincare routine"])
    
    return recommended, skin_tips[:4]

def validate_image(file):
    """
    Validate uploaded image file
    """
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    if file.filename == '':
        return False, "No file selected"
    
    if '.' not in file.filename:
        return False, "Invalid file format"
    
    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type not allowed. Please use: {', '.join(ALLOWED_EXTENSIONS)}"
    
    return True, "Valid file"