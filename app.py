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

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edged = cv2.Canny(blurred, 50, 150)

        # Dilation followed by erosion to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        signature_candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800:  # Skip small noise
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            # Signature usually wide and not very tall
            if 2.0 < aspect_ratio < 8.0 and h > 20 and y > img.shape[0] // 2:
                signature_candidates.append((cnt, area))

        if not signature_candidates:
            return jsonify({"error": "No signature-like region found"}), 400

        # Choose the largest among signature-like contours
        c = max(signature_candidates, key=lambda x: x[1])[0]
        x, y, w, h = cv2.boundingRect(c)
        signature_crop = img[y:y+h, x:x+w]

        # Convert to transparent PNG
        signature_rgba = cv2.cvtColor(signature_crop, cv2.COLOR_BGR2BGRA)

        # Make white pixels transparent
        white_mask = np.all(signature_rgba[:, :, :3] > [200, 200, 200], axis=-1)
        signature_rgba[white_mask, 3] = 0

        # Encode the result
        _, buffer = cv2.imencode('.png', signature_rgba)
        b64_output = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"signature_base64": b64_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
