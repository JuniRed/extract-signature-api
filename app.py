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

        # Convert to grayscale and blur to reduce noise
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find external contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area, aspect ratio, and vertical location
        signature_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # ignore small noise
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            if 2 < aspect_ratio < 10 and y > height * 0.5:
                signature_candidates.append((cnt, area))

        if not signature_candidates:
            return jsonify({"error": "No signature-like region found"}), 400

        # Choose the best candidate (by largest area)
        best_cnt = max(signature_candidates, key=lambda x: x[1])[0]
        x, y, w, h = cv2.boundingRect(best_cnt)
        signature_crop = img[y:y+h, x:x+w]

        # Convert to transparent PNG
        signature_rgba = cv2.cvtColor(signature_crop, cv2.COLOR_BGR2BGRA)
        white = np.all(signature_rgba[:, :, :3] == [255, 255, 255], axis=-1)
        signature_rgba[white, 3] = 0

        # Encode to PNG base64
        _, buffer = cv2.imencode('.png', signature_rgba)
        b64_output = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"signature_base64": b64_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
