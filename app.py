from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import numpy as np
import cv2

app = Flask(__name__)

def extract_signature_from_image(image_bytes):
    # Convert image bytes to OpenCV format
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest contour (likely signature)
    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)
    signature_crop = image_cv[y:y+h, x:x+w]

    # Convert cropped image back to PNG
    _, buffer = cv2.imencode('.png', signature_crop)
    signature_base64 = base64.b64encode(buffer).decode('utf-8')
    return signature_base64

@app.route('/extract_signature', methods=['POST'])
def extract_signature():
    data = request.get_json()
    file_base64 = data.get('file_base64', '')

    if not file_base64:
        return jsonify({'error': 'Missing file_base64'}), 400

    try:
        file_bytes = base64.b64decode(file_base64)
        signature_b64 = extract_signature_from_image(file_bytes)
        return jsonify({'signature_base64': signature_b64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
