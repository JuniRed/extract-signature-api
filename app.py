from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np

app = Flask(__name__)  # <----- must be defined first

@app.route('/extract_signature', methods=['POST'])
def extract_signature():
    data = request.get_json()
    base64_str = data.get("file_base64")

    if not base64_str:
        return jsonify({"error": "No file_base64 provided"}), 400

    try:
        # Decode base64 to image
        file_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 8
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        signature_contours = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(c)
            if 500 < area < 20000 and 2.0 < aspect_ratio < 6.0:
                signature_contours.append(c)

        if not signature_contours:
            return jsonify({"error": "No signature-like region found"}), 400

        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        for c in signature_contours:
            x, y, w, h = cv2.boundingRect(c)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        signature_crop = img[y_min:y_max, x_min:x_max]

        signature_rgba = cv2.cvtColor(signature_crop, cv2.COLOR_BGR2BGRA)
        white_mask = np.all(signature_rgba[:, :, :3] > [240, 240, 240], axis=-1)
        signature_rgba[white_mask, 3] = 0

        _, buffer = cv2.imencode('.png', signature_rgba)
        b64_output = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"signature_base64": b64_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
