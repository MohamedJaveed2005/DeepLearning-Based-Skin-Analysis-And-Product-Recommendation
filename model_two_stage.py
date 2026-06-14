import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

def create_skin_type_model():
    """
    Model 1: Skin Type Classifier (Oily/Dry/Normal)
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')  # 3 classes: oily, dry, normal
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_acne_model():
    """
    Model 2: Acne Detector (Acne/No Acne)
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(2, activation='softmax')  # 2 classes: acne, no_acne
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class TwoStageSkinAnalyzer:
    """
    Combined two-stage model for skin analysis
    """
    def __init__(self):
        self.skin_type_model = None
        self.acne_model = None
        self.skin_classes = ['Oily', 'Dry', 'Normal']
        self.acne_classes = ['No Acne', 'Acne Detected']
    
    def load_models(self, skin_type_path='models/skin_type_model.h5', 
                    acne_path='models/acne_model.h5'):
        """Load pre-trained models"""
        try:
            self.skin_type_model = keras.models.load_model(skin_type_path)
            self.acne_model = keras.models.load_model(acne_path)
            print("✅ Models loaded successfully")
            return True
        except:
            print("⚠️ Models not found. Please train first.")
            return False
    
    def preprocess_image(self, image_path):
        """Preprocess image for both models"""
        img = keras.preprocessing.image.load_img(
            image_path, target_size=(224, 224)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0
        return img_array
    
    def predict(self, image_path):
        """Predict both skin type and acne status"""
        img_array = self.preprocess_image(image_path)
        
        # Predict skin type
        skin_pred = self.skin_type_model.predict(img_array, verbose=0)[0]
        skin_type_idx = np.argmax(skin_pred)
        skin_type = self.skin_classes[skin_type_idx]
        skin_confidence = float(skin_pred[skin_type_idx])
        
        # Predict acne
        acne_pred = self.acne_model.predict(img_array, verbose=0)[0]
        acne_idx = np.argmax(acne_pred)
        acne_status = self.acne_classes[acne_idx]
        acne_confidence = float(acne_pred[acne_idx])
        
        return {
            'skin_type': skin_type,
            'skin_confidence': skin_confidence,
            'acne_status': acne_status,
            'acne_confidence': acne_confidence
        }

def train_skin_type_model():
    """Train the skin type classification model"""
    print("\n🔵 Training Skin Type Model...")
    print("="*50)
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load data
    train_generator = train_datagen.flow_from_directory(
        'dataset_skin_type/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        'dataset_skin_type/validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    # Create and train model
    model = create_skin_type_model()
    
    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        verbose=1
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/skin_type_model.h5')
    print("\n✅ Skin Type Model saved as 'models/skin_type_model.h5'")
    
    return history

def train_acne_model():
    """Train the acne detection model"""
    print("\n🔴 Training Acne Detection Model...")
    print("="*50)
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load data
    train_generator = train_datagen.flow_from_directory(
        'dataset_acne/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        'dataset_acne/validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    # Create and train model
    model = create_acne_model()
    
    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        verbose=1
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/acne_model.h5')
    print("\n✅ Acne Model saved as 'models/acne_model.h5'")
    
    return history

def train_all_models():
    """Train both models"""
    print("\n🚀 Starting Two-Stage Model Training")
    print("="*60)
    
    # Train skin type model
    skin_history = train_skin_type_model()
    
    # Train acne model
    acne_history = train_acne_model()
    
    print("\n" + "="*60)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    
    return skin_history, acne_history

if __name__ == "__main__":
    train_all_models()