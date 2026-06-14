import tensorflow as tf
import numpy as np
import os

def diagnose_skin_model():
    """Diagnose skin type model class order"""
    print("🔍 Diagnosing Skin Type Model...")
    print("="*50)
    
    # Load model
    try:
        model = tf.keras.models.load_model('models/skin_type_model.h5')
        print("✅ Skin model loaded")
    except:
        print("❌ Cannot load skin model")
        return
    
    # Check dataset structure
    train_path = 'dataset_skin_type/train'
    if os.path.exists(train_path):
        classes = sorted(os.listdir(train_path))
        print(f"\n📁 Training classes (alphabetical order):")
        for i, cls in enumerate(classes):
            print(f"   Index {i} = {cls}")
    else:
        print("❌ Dataset not found!")
    
    # Test on random input
    print(f"\n🧪 Testing on random input...")
    random_input = np.random.random((1, 224, 224, 3))
    pred = model.predict(random_input, verbose=0)[0]
    
    print(f"   Raw predictions: {pred}")
    print(f"   Index 0 (likely dry): {pred[0]:.4f}")
    print(f"   Index 1 (likely normal): {pred[1]:.4f}")
    print(f"   Index 2 (likely oily): {pred[2]:.4f}")
    
    highest_idx = np.argmax(pred)
    print(f"   Model defaults to index: {highest_idx}")

if __name__ == "__main__":
    diagnose_skin_model()