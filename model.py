import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import random

def create_skin_analysis_model():
    """
    Creates a deep learning model for skin analysis
    """
    # Simple CNN model for demonstration
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(5, activation='softmax')  # 3 skin types + 2 acne statuses
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(image_path):
    """
    Preprocess image for model prediction
    """
    img = keras.preprocessing.image.load_img(
        image_path, 
        target_size=(224, 224)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    
    return img_array

def predict_skin_conditions(model, image_path):
    """
    Predict skin type and acne presence
    For demo, uses random prediction (replace with real model later)
    """
    # For demonstration - replace with real model prediction
    # Comment out the random part and uncomment the model prediction when you have trained model
    
    # Mock prediction for testing
    skin_types = ['Oily', 'Dry', 'Normal']
    acne_types = ['No Acne', 'Acne Detected']
    
    # Random prediction for demo
    skin_type = random.choice(skin_types)
    acne_status = random.choice(acne_types)
    
    # Uncomment below when you have a trained model
    """
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)[0]
    
    skin_type_idx = tf.argmax(predictions[:3]).numpy()
    skin_type = skin_types[skin_type_idx]
    
    acne_idx = tf.argmax(predictions[3:]).numpy()
    acne_status = acne_types[acne_idx]
    """
    
    return skin_type, acne_status