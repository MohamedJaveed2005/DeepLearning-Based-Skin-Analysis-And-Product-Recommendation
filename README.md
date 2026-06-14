# 🌿 Deep Learning Based Facial Skin Analysis and Product Recommendation System

A web-based intelligent skincare assistant that uses **Deep Learning**, **Computer Vision**, and **Flask** to analyze facial skin from an uploaded image, identify skin type and acne presence, and recommend suitable skincare products.

---

## 📖 Project Overview

This project leverages **Transfer Learning with MobileNetV2** to perform facial skin analysis. Users can upload a facial image, and the system automatically:

* Detects and crops the face
* Predicts skin type (Oily, Dry, Normal)
* Detects acne presence
* Generates skincare tips
* Recommends suitable skincare products

The application provides a modern wellness-themed user interface and secure user authentication.

---

## 🚀 Features

### 🔐 User Authentication

* User Registration (Signup)
* Secure Login System
* Password Hashing using Werkzeug
* Session Management using Flask-Login
* Protected Routes
* Logout Functionality

### 📸 Facial Image Upload

* Upload JPG, JPEG, PNG images
* Drag-and-drop support
* Image Preview
* Maximum file size: 16 MB

### 😀 Face Detection

* OpenCV Haar Cascade Classifier
* Automatic face detection and cropping
* Removes unnecessary background

### 🧠 Skin Type Classification

Predicts one of the following skin types:

* Dry Skin
* Normal Skin
* Oily Skin

Displays prediction confidence score.

### 🔍 Acne Detection

Binary Classification:

* Acne Detected
* No Acne

Displays confidence score for prediction.

### 🛍️ Product Recommendation Engine

Rule-based recommendation system that:

* Matches products to skin type
* Prioritizes acne-treatment products
* Recommends 3–5 suitable products

Each recommendation includes:

* Product Name
* Brand
* Category
* Price
* Description
* Product Image

### 💡 Personalized Skincare Tips

Provides customized skincare suggestions based on detected skin type.

### 🎨 Modern UI Design

Organic Spa & Wellness inspired interface featuring:

* Earthy green color palette
* Cream backgrounds
* Playfair Display typography
* Floating leaf animations
* Responsive design
* Interactive product cards

---

## 🏗️ System Architecture

```text
User Login
     │
     ▼
Upload Facial Image
     │
     ▼
Face Detection (OpenCV)
     │
     ▼
Image Preprocessing
     │
     ▼
 ┌─────────────────────┐
 │ Skin Type Model     │
 └─────────────────────┘
     │
     ▼
 ┌─────────────────────┐
 │ Acne Detection Model│
 └─────────────────────┘
     │
     ▼
Recommendation Engine
     │
     ▼
Results + Tips + Products
```

---

## 🛠️ Technologies Used

| Layer             | Technology              |
| ----------------- | ----------------------- |
| Frontend          | HTML5, CSS3, JavaScript |
| Backend           | Flask                   |
| Deep Learning     | TensorFlow, Keras       |
| Transfer Learning | MobileNetV2             |
| Computer Vision   | OpenCV                  |
| Authentication    | Flask-Login             |
| Database          | SQLite                  |
| ORM               | Flask-SQLAlchemy        |
| Product Storage   | JSON                    |
| Deployment Ready  | Flask                   |

---

## 🧠 Deep Learning Models

### 1️⃣ Skin Type Classification Model

**Architecture**

* MobileNetV2 (Pretrained on ImageNet)
* Global Average Pooling Layer
* Dense Layers
* Softmax Output Layer

**Input**

```text
224 × 224 × 3 RGB Image
```

**Output Classes**

```text
Dry
Normal
Oily
```

**Accuracy**

```text
~85%
```

---

### 2️⃣ Acne Detection Model

**Architecture**

* MobileNetV2 (Transfer Learning)
* Custom Dense Layers
* Binary Classification Output

**Input**

```text
224 × 224 × 3 RGB Image
```

**Output Classes**

```text
Acne Detected
No Acne
```

**Accuracy**

```text
~90%
```

---

## 📊 Datasets Used

### Skin Type Dataset

* Source: Kaggle
* Author: Shakya Dissanayake
* Images: 3,153

Classes:

* Dry
* Normal
* Oily

### Acne Dataset

* Source: Kaggle
* Author: Nayan Chaure
* Images: 1,833

Classes:

* Acne
* No Acne

---

## 🔄 Workflow

### Step 1

User logs into the application.

### Step 2

User uploads a facial image.

### Step 3

OpenCV detects and crops the face.

### Step 4

Image is resized to:

```text
224 × 224
```

and normalized.

### Step 5

Skin Type Model predicts:

* Dry
* Normal
* Oily

### Step 6

Acne Detection Model predicts:

* Acne Detected
* No Acne

### Step 7

Recommendation Engine generates personalized products.

### Step 8

Results page displays:

* Uploaded Image
* Skin Type
* Acne Status
* Confidence Scores
* Recommended Products
* Skincare Tips

---

## 📂 Project Structure

```text
DeepLearning-Based-Skin-Analysis-And-Product-Recommendation/
│
├── static/
│   ├── css/
│   ├── js/
│   ├── uploads/
│   └── images/
│
├── templates/
│   ├── login.html
│   ├── signup.html
│   ├── dashboard.html
│   └── result.html
│
├── models/
│   ├── skin_type_model.h5
│   └── acne_model.h5
│
├── products.json
├── app.py
├── requirements.txt
├── users.db
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/MohamedJaveed2005/DeepLearning-Based-Skin-Analysis-And-Product-Recommendation.git

cd DeepLearning-Based-Skin-Analysis-And-Product-Recommendation
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

## 📸 Sample Output

### Analysis Result

```text
Skin Type: Oily
Confidence: 92.4%

Acne Status: Detected
Confidence: 89.8%
```

### Recommended Products

```text
✔ Salicylic Acid Face Wash
✔ Oil-Free Moisturizer
✔ Acne Control Serum
✔ Sunscreen SPF 50
```

---

## 🔮 Future Enhancements

* Real-time webcam analysis
* Skin tone detection
* Wrinkle and pigmentation detection
* AI chatbot skincare assistant
* Product recommendation using Machine Learning
* Cloud deployment on AWS
* Mobile application version
* Multi-language support

---

## 👨‍💻 Author

**Mohamed Javeed A**

Software Engineer | Deep Learning Enthusiast | Cloud Computing Enthusiast

GitHub:
https://github.com/MohamedJaveed2005

---

## ⭐ Support

If you found this project useful, please consider giving it a **Star ⭐** on GitHub.

Your support helps motivate future improvements and open-source contributions.
