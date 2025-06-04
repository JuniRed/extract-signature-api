from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os
from pdf2image import convert_from_bytes

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

        # Decode as image or PDF
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            try:
                pil_images = convert_from_bytes(file_bytes)
                img = cv2.cvtColor(np.array(pil_images[0]), cv2.COLOR_RGB2BGR)
            except Exception as e:
                return jsonify({"error": "Could not decode as image or PDF: " + str(e)}), 400

        # Convert to grayscale and blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Edge detection for handwriting-like strokes
        edges = cv2.Canny(blurred, 50, 150)

        # Morph to connect signature strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = filter_signature_like_contours(contours, gray.shape)

        if not filtered:
            return jsonify({"error": "No signature found"}), 400

        # Extract all signature-like areas
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, filtered, -1, 255, -1)

        x, y, w, h = cv2.boundingRect(np.vstack(filtered))
        cropped = img[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]

        # Apply transparency
        result_rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        result_rgba[cropped_mask == 0, 3] = 0

        _, buffer = cv2.imencode('.png', result_rgba)
        b64_output = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "signature_base64": b64_output,
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "contour_count": len(filtered)
        })

    except Exception as e:
        return jsonify({"error": "Processing failed: " + str(e)}), 500


def filter_signature_like_contours(contours, image_shape):
    filtered = []
    img_h, img_w = image_shape
    img_area = img_w * img_h

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        length = cv2.arcLength(cnt, True)
        aspect_ratio = w / float(h) if h > 0 else 0
        solidity = area / float(w * h + 1e-5)

        # Signature: long, low-solid, irregular strokes
        if (
            500 < area < 0.05 * img_area and
            3 < aspect_ratio < 15 and
            0.05 < solidity < 0.4 and
            length > 100
        ):
            filtered.append(cnt)
    return filtered

if __name__ == '__main__':
    app.run(debug=True)
