# Create check_training_data.py
import os

def check_class_balance():
    """Check what's actually in your training folders"""
    
    train_acne = 'dataset_acne/train/acne'
    train_no_acne = 'dataset_acne/train/no_acne'
    
    if os.path.exists(train_acne):
        acne_count = len(os.listdir(train_acne))
        print(f"✅ Acne folder: {acne_count} images")
        
        # Sample check first 5 images
        print("\n📁 Sample acne images:")
        for img in os.listdir(train_acne)[:5]:
            print(f"   - {img}")
    else:
        print("❌ Acne folder not found!")
    
    if os.path.exists(train_no_acne):
        no_acne_count = len(os.listdir(train_no_acne))
        print(f"\n✅ No Acne folder: {no_acne_count} images")
        
        # Sample check first 5 images
        print("\n📁 Sample no_acne images:")
        for img in os.listdir(train_no_acne)[:5]:
            print(f"   - {img}")
    else:
        print("❌ No Acne folder not found!")
    
    print(f"\n📊 Class balance: {acne_count} vs {no_acne_count}")

if __name__ == "__main__":
    check_class_balance()