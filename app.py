from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf
import io
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ----------------------------------------------------------------
# 1) Load Model
# ----------------------------------------------------------------
try:
    model_path = r"microplastic detector and classfier.h5"
    logger.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

class_labels = ["Microplastic", "Air Bubble", "Organic Matter"]

# ----------------------------------------------------------------
# 2) Preprocess (as in your notebook)
# ----------------------------------------------------------------
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read {image_path}")
        return None, None

    # LAB + CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    contrast_enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Sharpen
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharp_image = cv2.filter2D(contrast_enhanced_image, -1, sharpen_kernel)

    # Bilateral Filter
    denoised_image = cv2.bilateralFilter(sharp_image, d=9, sigmaColor=75, sigmaSpace=75)

    # Normalized
    normalized_image = denoised_image / 255.0

    return image, normalized_image

# ----------------------------------------------------------------
# 3) Detect Bounding Boxes (as in your notebook)
# ----------------------------------------------------------------
def detect_bounding_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    min_area = 500
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > min_area:
            bounding_boxes.append((x, y, w, h))
    return bounding_boxes

# ----------------------------------------------------------------
# 4) Preprocess ROI for Model
# ----------------------------------------------------------------
def preprocess_roi(roi, img_size=224):
    roi = cv2.resize(roi, (img_size, img_size))
    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)
    return roi

# ----------------------------------------------------------------
# 5) LABEL MAPPING to Swap Classes
# ----------------------------------------------------------------
# If your model is mixing up Microplastic ↔ Air Bubble,
# define a mapping to correct it in the final output.
label_mapping = {
    "Microplastic": "Air Bubble",  # The model says Microplastic, we want to label it as Air Bubble
    "Air Bubble": "Microplastic",  # The model says Air Bubble, we want to label it as Microplastic
    "Organic Matter": "Organic Matter"
}

color_map = {
    "Microplastic": (0, 255, 0),   # Green
    "Air Bubble": (255, 0, 0),     # Blue
    "Organic Matter": (0, 0, 255)  # Red
}

# ----------------------------------------------------------------
# 6) Flask Endpoint
# ----------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']
        filename = secure_filename(file.filename)
        input_path = f"temp_{filename}"
        file.save(input_path)

        original_image, _ = preprocess_image(input_path)
        if original_image is None:
            return jsonify({"error": "Failed to process image"}), 500

        bounding_boxes = detect_bounding_boxes(original_image)
        logger.info(f"✅ Found {len(bounding_boxes)} objects.")

        for (x, y, w, h) in bounding_boxes:
            roi = original_image[y:y+h, x:x+w]
            roi_preprocessed = preprocess_roi(roi)

            # Predict
            prediction = model.predict(roi_preprocessed)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            raw_label = class_labels[predicted_class]

            # Swap classes if needed
            final_label = label_mapping[raw_label]

            # Draw
            color = color_map[final_label]
            cv2.rectangle(original_image, (x, y), (x + w, y + h), color, 2)
            text = f"{final_label} ({confidence:.2f}%)"
            cv2.putText(
                original_image,
                text,
                (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        _, img_encoded = cv2.imencode('.png', original_image)
        os.remove(input_path)

        return send_file(
            io.BytesIO(img_encoded.tobytes()),
            mimetype='image/png',
            as_attachment=True,
            download_name='annotated.png'
        )

    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
