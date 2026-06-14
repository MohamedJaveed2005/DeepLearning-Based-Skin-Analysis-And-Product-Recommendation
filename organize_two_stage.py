import os
import shutil
import random
from pathlib import Path

def organize_two_stage_datasets():
    """
    Organize datasets for two-stage model approach
    """
    
    # Paths - UPDATE THESE to match your extracted folders
    skin_type_source = 'Oily-Dry-Skin-Types'  # Your skin type dataset folder
    acne_source = 'Acne'  # Your acne dataset folder
    
    # Create directory structures
    os.makedirs('dataset_skin_type/train', exist_ok=True)
    os.makedirs('dataset_skin_type/validation', exist_ok=True)
    os.makedirs('dataset_acne/train', exist_ok=True)
    os.makedirs('dataset_acne/validation', exist_ok=True)
    
    print("📁 Creating dataset structure...")
    
    # ============================================
    # PART 1: Organize Skin Type Dataset
    # ============================================
    print("\n🔵 Processing Skin Type Dataset...")
    
    skin_classes = ['oily', 'dry', 'normal']
    
    for cls in skin_classes:
        os.makedirs(f'dataset_skin_type/train/{cls}', exist_ok=True)
        os.makedirs(f'dataset_skin_type/validation/{cls}', exist_ok=True)
    
    # Copy from existing train/validation split
    for split in ['train', 'valid']:  # Your dataset uses 'valid' not 'validation'
        source_base = os.path.join(skin_type_source, split)
        dest_base = 'dataset_skin_type/train' if split == 'train' else 'dataset_skin_type/validation'
        
        for cls in skin_classes:
            source_folder = os.path.join(source_base, cls)
            dest_folder = os.path.join(dest_base, cls)
            
            if os.path.exists(source_folder):
                for img_file in os.listdir(source_folder):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src = os.path.join(source_folder, img_file)
                        dst = os.path.join(dest_folder, img_file)
                        shutil.copy2(src, dst)
                
                count = len(os.listdir(dest_folder))
                print(f"  ✅ {split}/{cls}: {count} images")
    
    # ============================================
    # PART 2: Organize Acne Dataset
    # ============================================
    print("\n🔴 Processing Acne Dataset...")
    
    os.makedirs('dataset_acne/train/acne', exist_ok=True)
    os.makedirs('dataset_acne/train/no_acne', exist_ok=True)
    os.makedirs('dataset_acne/validation/acne', exist_ok=True)
    os.makedirs('dataset_acne/validation/no_acne', exist_ok=True)
    
    # Copy acne images
    acne_images = []
    for root, dirs, files in os.walk(acne_source):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                acne_images.append(os.path.join(root, file))
    
    print(f"  Found {len(acne_images)} acne images")
    
    # Shuffle and split (80% train, 20% validation)
    random.shuffle(acne_images)
    split_idx = int(len(acne_images) * 0.8)
    train_acne = acne_images[:split_idx]
    valid_acne = acne_images[split_idx:]
    
    # Copy acne images
    for img_path in train_acne:
        filename = os.path.basename(img_path)
        dst = os.path.join('dataset_acne/train/acne', filename)
        shutil.copy2(img_path, dst)
    
    for img_path in valid_acne:
        filename = os.path.basename(img_path)
        dst = os.path.join('dataset_acne/validation/acne', filename)
        shutil.copy2(img_path, dst)
    
    print(f"  ✅ train/acne: {len(train_acne)} images")
    print(f"  ✅ validation/acne: {len(valid_acne)} images")
    
    # ============================================
    # PART 3: Create "No Acne" dataset
    # ============================================
    print("\n🟢 Creating 'No Acne' dataset from skin type images...")
    
    # Use a portion of skin type images as "no acne" samples
    # This assumes skin type dataset contains mostly clear skin
    
    no_acne_images = []
    for split in ['train', 'valid']:
        source_base = os.path.join(skin_type_source, split)
        for cls in skin_classes:
            source_folder = os.path.join(source_base, cls)
            if os.path.exists(source_folder):
                for img_file in os.listdir(source_folder):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        no_acne_images.append(os.path.join(source_folder, img_file))
    
    # Shuffle and select same number as acne images (for balance)
    random.shuffle(no_acne_images)
    no_acne_images = no_acne_images[:len(acne_images)]
    
    # Split (80% train, 20% validation)
    split_idx = int(len(no_acne_images) * 0.8)
    train_no_acne = no_acne_images[:split_idx]
    valid_no_acne = no_acne_images[split_idx:]
    
    # Copy no_acne images
    for img_path in train_no_acne:
        filename = os.path.basename(img_path)
        dst = os.path.join('dataset_acne/train/no_acne', filename)
        shutil.copy2(img_path, dst)
    
    for img_path in valid_no_acne:
        filename = os.path.basename(img_path)
        dst = os.path.join('dataset_acne/validation/no_acne', filename)
        shutil.copy2(img_path, dst)
    
    print(f"  ✅ train/no_acne: {len(train_no_acne)} images")
    print(f"  ✅ validation/no_acne: {len(valid_no_acne)} images")
    
    # ============================================
    # Summary
    # ============================================
    print("\n" + "="*50)
    print("📊 DATASET ORGANIZATION COMPLETE!")
    print("="*50)
    
    print("\n🔵 SKIN TYPE DATASET:")
    for cls in skin_classes:
        train_count = len(os.listdir(f'dataset_skin_type/train/{cls}'))
        valid_count = len(os.listdir(f'dataset_skin_type/validation/{cls}'))
        print(f"  {cls}: Train={train_count}, Valid={valid_count}")
    
    print("\n🔴 ACNE DATASET:")
    for cls in ['acne', 'no_acne']:
        train_count = len(os.listdir(f'dataset_acne/train/{cls}'))
        valid_count = len(os.listdir(f'dataset_acne/validation/{cls}'))
        print(f"  {cls}: Train={train_count}, Valid={valid_count}")
    
    print("\n✅ Ready for training!")

if __name__ == "__main__":
    organize_two_stage_datasets()