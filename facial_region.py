import cv2
import os


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def extract_regions(image):
    """
    Detect face and divide it into
    forehead, nose, cheeks and chin.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) == 0:
        return None

    # Take the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    face = image[y:y+h, x:x+w]

    # ------------------------
    # Region Coordinates
    # ------------------------

    
    forehead = face[
    int(h * 0.08):int(h * 0.30),
    int(w * 0.15):int(w * 0.85)
    ]

    nose = face[
        int(h * 0.30):int(h * 0.65),
        int(w * 0.30):int(w * 0.70)
    ]

    left_cheek = face[
    int(h * 0.35):int(h * 0.75),
    int(w * 0.05):int(w * 0.38)
    ]

    right_cheek = face[
    int(h * 0.35):int(h * 0.75),
    int(w * 0.62):int(w * 0.95)
    ]

    chin = face[
        int(h * 0.75):h,
        int(w * 0.20):int(w * 0.80)
    ]

    return {
        "face": face,
        "bbox": (x, y, w, h),
        "forehead": forehead,
        "nose": nose,
        "left_cheek": left_cheek,
        "right_cheek": right_cheek,
        "chin": chin
    }


def save_regions(regions, save_folder="static/region_images"):
    """
    Save all cropped facial regions.
    """

    os.makedirs(save_folder, exist_ok=True)

    cv2.imwrite(
        os.path.join(save_folder, "forehead.jpg"),
        regions["forehead"]
    )

    cv2.imwrite(
        os.path.join(save_folder, "nose.jpg"),
        regions["nose"]
    )

    cv2.imwrite(
        os.path.join(save_folder, "left_cheek.jpg"),
        regions["left_cheek"]
    )

    cv2.imwrite(
        os.path.join(save_folder, "right_cheek.jpg"),
        regions["right_cheek"]
    )

    cv2.imwrite(
        os.path.join(save_folder, "chin.jpg"),
        regions["chin"]
    )


def draw_regions(image, bbox):
    """
    Draw rectangles around facial regions.
    """

    x, y, w, h = bbox

    output = image.copy()

    # Forehead
    cv2.rectangle(
        output,
        (x, y),
        (x+w, y+int(h*0.25)),
        (0,255,0),
        2
    )

    # Nose
    cv2.rectangle(
        output,
        (x+int(w*0.30), y+int(h*0.30)),
        (x+int(w*0.70), y+int(h*0.65)),
        (255,0,0),
        2
    )

    # Left Cheek
    cv2.rectangle(
        output,
        (x, y+int(h*0.35)),
        (x+int(w*0.30), y+int(h*0.75)),
        (0,0,255),
        2
    )

    # Right Cheek
    cv2.rectangle(
        output,
        (x+int(w*0.70), y+int(h*0.35)),
        (x+w, y+int(h*0.75)),
        (255,255,0),
        2
    )

    # Chin
    cv2.rectangle(
        output,
        (x+int(w*0.20), y+int(h*0.75)),
        (x+int(w*0.80), y+h),
        (255,0,255),
        2
    )

    return output