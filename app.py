from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/extract_signature', methods=['POST'])
def extract_signature():
    data = request.get_json()
    base64_str = data.get("file_base64")

    if not base64_str:
        return jsonify({"error": "No file_base64 provided"}), 400

    try:
        file_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Convert to grayscale and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get largest contour
        if not contours:
            return jsonify({"error": "No signature found"}), 400

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        signature_crop = img[y:y+h, x:x+w]

        # Convert to RGBA
        signature_rgba = cv2.cvtColor(signature_crop, cv2.COLOR_BGR2BGRA)

        # Make white background transparent
        white = np.all(signature_rgba[:, :, :3] == [255, 255, 255], axis=-1)
        signature_rgba[white, 3] = 0

        # Convert to PNG
        _, buffer = cv2.imencode('.png', signature_rgba)
        b64_output = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"signature_base64": b64_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
