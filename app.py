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

        h_img, w_img = img.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur and thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        signature_candidates = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            # Area filtering
            if area < 1000 or area > 0.3 * (h_img * w_img):
                continue

            # Bounding box shape filtering
            if w > 0.9 * w_img or h > 0.3 * h_img:
                continue

            aspect_ratio = w / float(h)
            if aspect_ratio < 2 or aspect_ratio > 8:
                continue

            # Position: bottom half of image
            if y < h_img * 0.3:
                continue

            # Solidity check (avoid logos/stamps)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            non_zero = cv2.countNonZero(mask[y:y+h, x:x+w])
            rect_area = w * h
            solidity = non_zero / float(rect_area)

            if solidity < 0.2 or solidity > 0.9:
                continue

            signature_candidates.append((cnt, area))

        if not signature_candidates:
            return jsonify({"error": "No signature-like region found"}), 400

        # Choose largest valid signature candidate
        signature_cnt = max(signature_candidates, key=lambda x: x[1])[0]
        x, y, w, h = cv2.boundingRect(signature_cnt)
        signature_crop = img[y:y+h, x:x+w]

        # Convert cropped image to transparent PNG
        signature_rgba = cv2.cvtColor(signature_crop, cv2.COLOR_BGR2BGRA)
        white_pixels = np.all(signature_rgba[:, :, :3] >= [245, 245, 245], axis=-1)
        signature_rgba[white_pixels, 3] = 0

        # Encode result
        _, buffer = cv2.imencode('.png', signature_rgba)
        b64_output = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"signature_base64": b64_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=10000)
