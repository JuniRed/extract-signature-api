from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/extract_signature', methods=['POST'])
def extract_signature():
    try:
        data = request.get_json()
        base64_str = data.get("file_base64")
        if not base64_str:
            return jsonify({"error": "No file_base64 provided"}), 400

        file_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological filter to connect handwriting strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        height = img.shape[0]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = w / float(h)

            if area < 800 or area > 70000:
                continue
            if aspect_ratio < 2.5 or aspect_ratio > 12:
                continue
            if y < height * 0.4:
                continue  # likely not at bottom

            roi = gray[y:y+h, x:x+w]
            stroke_density = np.count_nonzero(roi < 128) / (w * h)
            if stroke_density < 0.03 or stroke_density > 0.5:
                continue

            candidates.append((cnt, area))

        if not candidates:
            return jsonify({"error": "No signature-like region found"}), 400

        best_cnt, _ = max(candidates, key=lambda x: x[1])
        x, y, w, h = cv2.boundingRect(best_cnt)
        signature_crop = img[y:y+h, x:x+w]

        # Remove white background
        signature_rgba = cv2.cvtColor(signature_crop, cv2.COLOR_BGR2BGRA)
        white = np.all(signature_rgba[:, :, :3] > [200, 200, 200], axis=-1)
        signature_rgba[white, 3] = 0

        _, buffer = cv2.imencode('.png', signature_rgba)
        b64_output = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"signature_base64": b64_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

