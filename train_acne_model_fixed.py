import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json
import os

def create_acne_model():
    """Create acne detection model"""
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
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_fixed_acne_model():
    """Train and save model with class labels"""
    print("\n🔴 Training Acne Model with Fixed Labels...")
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
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        'dataset_acne/validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Print class indices
    print(f"\n📋 Class indices: {train_generator.class_indices}")
    
    # Save class mapping
    class_mapping = train_generator.class_indices
    with open('models/acne_class_labels.json', 'w') as f:
        json.dump(class_mapping, f)
    print(f"✅ Saved class mapping: {class_mapping}")
    
    # Create and train model
    model = create_acne_model()
    
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        verbose=1
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/acne_model_fixed.h5')
    print("\n✅ Model saved as 'models/acne_model_fixed.h5'")
    
    # Test prediction on a few samples
    print("\n🧪 Testing predictions...")
    test_predictions(model, validation_generator, class_mapping)
    
    return model, class_mapping

def test_predictions(model, generator, class_mapping):
    """Test model predictions to verify class order"""
    # Get a batch of images
    images, labels = next(generator)
    
    # Predict
    predictions = model.predict(images, verbose=0)
    
    # Reverse mapping (index -> class name)
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    print("\n📊 Sample Predictions:")
    for i in range(min(5, len(images))):
        true_idx = np.argmax(labels[i])
        pred_idx = np.argmax(predictions[i])
        confidence = predictions[i][pred_idx]
        
        true_class = idx_to_class[true_idx]
        pred_class = idx_to_class[pred_idx]
        
        status = "✅" if true_idx == pred_idx else "❌"
        print(f"  {status} True: {true_class:8s} | Pred: {pred_class:8s} | Conf: {confidence:.2%}")

if __name__ == "__main__":
    train_fixed_acne_model()