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
            return jsonify({"error": "Missing file_base64"}), 400

        file_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        signature_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 3000 < area < 50000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if aspect_ratio > 2.5:
                    signature_candidates.append((cnt, area))

        signature_candidates = sorted(signature_candidates, key=lambda x: x[1], reverse=True)

        if not signature_candidates:
            return jsonify({"error": "No signature-like region found"}), 400

        best_cnt, _ = signature_candidates[0]
        x, y, w, h = cv2.boundingRect(best_cnt)
        signature_crop = img[y:y + h, x:x + w]

        signature_rgba = cv2.cvtColor(signature_crop, cv2.COLOR_BGR2BGRA)
        white = np.all(signature_rgba[:, :, :3] == [255, 255, 255], axis=-1)
        signature_rgba[white, 3] = 0

        _, buffer = cv2.imencode('.png', signature_rgba)
        b64_output = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"signature_base64": b64_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
