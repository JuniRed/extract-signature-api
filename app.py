from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/extract_signature', methods=['POST'])
def extract_signature():
    data = request.get_json()
    base64_str = data.get("file_base64")
    if not base64_str:
        return jsonify({"error": "No file_base64 provided"}), 400

    try:
        # Decode base64 image
        file_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        height, width = img.shape[:2]

        # Focus on the bottom half where signatures usually appear
        bottom_half = img[height // 2:, :]

        # Convert to grayscale
        gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 4
        )

        # Morphological operations to enhance signature features
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter potential signature contours
        signature_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            if area > 500 and 1.5 < aspect_ratio < 10 and h > 15:
                signature_contours.append((x, y, w, h))

        if not signature_contours:
            return jsonify({"error": "No signature-like region found"}), 400

        # Choose the largest by area
        best_x, best_y, best_w, best_h = max(signature_contours, key=lambda b: b[2]*b[3])

        # Adjust Y since we sliced only bottom half
        best_y += height // 2

        # Crop signature
        signature_crop = img[best_y:best_y+best_h, best_x:best_x+best_w]

        # Convert to transparent PNG (remove white background)
        signature_rgba = cv2.cvtColor(signature_crop, cv2.COLOR_BGR2BGRA)
        white = np.all(signature_rgba[:, :, :3] >= [240, 240, 240], axis=-1)
        signature_rgba[white, 3] = 0

        # Encode as base64 PNG
        _, buffer = cv2.imencode('.png', signature_rgba)
        b64_output = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"signature_base64": b64_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
