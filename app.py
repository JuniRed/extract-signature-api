from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import tempfile
import os
from pdf2image import convert_from_bytes
from PIL import Image

app = Flask(__name__)

def pdf_to_image(base64_str):
    pdf_bytes = base64.b64decode(base64_str)
    images = convert_from_bytes(pdf_bytes, dpi=300)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        images[0].save(temp.name, format="PNG")
        return cv2.imread(temp.name)

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply slight blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold to isolate ink
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )
    return thresh

def find_signature_contour(thresh, img_height):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    signature_candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # Filter based on area and position
        if 1000 < area < 50000 and aspect_ratio > 2 and y > img_height // 3:
            signature_candidates.append((cnt, area))

    if not signature_candidates:
        return None

    # Choose contour with the largest area
    return max(signature_candidates, key=lambda x: x[1])[0]

def extract_signature_image(img, contour):
    x, y, w, h = cv2.boundingRect(contour)
    signature = img[y:y+h, x:x+w]
    
    # Clean to white background
    gray = cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    b, g, r = cv2.split(signature)
    rgba = [b, g, r, alpha]
    signature_rgba = cv2.merge(rgba)

    # Fill background as white
    white_bg = np.ones_like(signature_rgba) * 255
    signature_cleaned = np.where(signature_rgba[:, :, 3:] == 0, white_bg, signature_rgba)

    return signature_cleaned.astype(np.uint8)

@app.route('/extract_signature', methods=['POST'])
def extract_signature():
    data = request.get_json()
    base64_str = data.get("file_base64")
    if not base64_str:
        return jsonify({"error": "No file_base64 provided"}), 400

    try:
        if base64_str.startswith("JVBER"):  # PDF magic header in base64
            img = pdf_to_image(base64_str)
        else:
            file_bytes = base64.b64decode(base64_str)
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Could not decode input"}), 400

        thresh = preprocess_image(img)
        contour = find_signature_contour(thresh, img.shape[0])

        if contour is None:
            return jsonify({"error": "No signature-like region found"}), 400

        signature = extract_signature_image(img, contour)
        _, buffer = cv2.imencode('.png', signature)
        signature_b64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({"signature_base64": signature_b64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
