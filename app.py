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
        # Decode base64 to image
        file_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400
        # Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Improve contrast
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        # Morph open + close to clean noise and connect ink
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        signature_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = w / float(h) if h > 0 else 0
            solidity = area / (w * h + 1e-5)  # prevent div by 0
            # Relaxed rules: area, aspect ratio, and ink density
            if area > 800 and 0.5 < aspect_ratio < 12 and solidity > 0.1:
                signature_contours.append(cnt)
        if not signature_contours:
            return jsonify({"error": "No signature-like region found"}), 400
        # Get the largest signature contour
        c = max(signature_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        signature_crop = img[y:y+h, x:x+w]
        # Convert to transparent PNG
        signature_rgba = cv2.cvtColor(signature_crop, cv2.COLOR_BGR2BGRA)
        white_mask = np.all(signature_rgba[:, :, :3] > 240, axis=-1)
        signature_rgba[white_mask, 3] = 0
        # Encode result
        _, buffer = cv2.imencode('.png', signature_rgba)
        b64_output = base64.b64encode(buffer).decode('utf-8')
        return jsonify({"signature_base64": b64_output})
    except Exception as e:
        return jsonify({"error": "Processing failed: " + str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
