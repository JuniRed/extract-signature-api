from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

def extract_signature_from_pdf(pdf_bytes):
    images = convert_from_bytes(pdf_bytes)
    image = np.array(images[0])  # only first page

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h < 150:  # adjust as needed
            cropped = image[y:y+h, x:x+w]
            signature = Image.fromarray(cropped)
            buffer = io.BytesIO()
            signature.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()

    return None

@app.route("/extract_signature", methods=["POST"])
def extract_signature():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    base64_img = extract_signature_from_pdf(file.read())

    if base64_img:
        return jsonify({"signature_image": base64_img})
    return jsonify({"error": "No signature found"}), 404

if __name__ == "__main__":
    app.run()
