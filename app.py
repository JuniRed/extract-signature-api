from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os
from pdf2image import convert_from_bytes
from PIL import Image
from io import BytesIO

app = Flask(__name__)

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

@app.route('/extract_signature', methods=['POST'])
def extract_signature():
    try:
        data = request.get_json()
        base64_str = data.get("file_base64")
        if not base64_str:
            return jsonify({"error": "No file_base64 provided"}), 400

        file_bytes = base64.b64decode(base64_str)

        # Try decoding as an image first
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # If image decode fails, try treating it as PDF
        if img is None:
            try:
                pil_images = convert_from_bytes(file_bytes)
                img_pil = pil_images[0]  # use the first page only
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                return jsonify({"error": "Could not decode as image or PDF: " + str(e)}), 400

        # Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        signature_contours = filter_contours(contours, tight=True)
        if not signature_contours:
            signature_contours = filter_contours(contours, tight=False)

        if not signature_contours:
            return jsonify({"error": "No signature-like region found"}), 400

        # Choose the largest contour
        c = max(signature_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        signature_crop = img[y:y + h, x:x + w]

        # Transparent background
        signature_rgba = cv2.cvtColor(signature_crop, cv2.COLOR_BGR2BGRA)
        white_mask = np.all(signature_rgba[:, :, :3] > 240, axis=-1)
        signature_rgba[white_mask, 3] = 0

        _, buffer = cv2.imencode('.png', signature_rgba)
        b64_output = base64.b64encode(buffer).decode('utf-8')

        if DEBUG_MODE:
            return jsonify({
                "signature_base64": b64_output,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "contour_count": len(contours)
            })

        return jsonify({"signature_base64": b64_output})

    except Exception as e:
        return jsonify({"error": "Processing failed: " + str(e)}), 500


def filter_contours(contours, tight=True):
    result = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = w / float(h) if h > 0 else 0
        solidity = area / float(w * h + 1e-5)

        if tight:
            if area > 1000 and 0.5 < aspect_ratio < 10 and solidity > 0.15:
                result.append(cnt)
        else:
            if area > 500 and 0.2 < aspect_ratio < 15 and solidity > 0.08:
                result.append(cnt)
    return result


if __name__ == '__main__':
    app.run(debug=True)
