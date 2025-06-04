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

        # Try decoding as image first
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # If not an image, try decoding as PDF
        if img is None:
            try:
                pil_images = convert_from_bytes(file_bytes)
                img_pil = pil_images[0]
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                return jsonify({"error": "Invalid file: " + str(e)}), 400

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Morphological filter to connect signature strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter signature-like contours
        img_h, img_w = gray.shape
        img_area = img_w * img_h
        signature_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = w / float(h) if h > 0 else 0
            solidity = area / float(w * h + 1e-5)
            length = cv2.arcLength(cnt, True)

            # Signature-like filter: thin, long, low fill, enough complexity
            if (500 < area < 0.05 * img_area and
                2 < aspect_ratio < 15 and
                0.05 < solidity < 0.5 and
                length > 100):
                signature_contours.append(cnt)

        if not signature_contours:
            return jsonify({"error": "No signature found"}), 400

        # Mask and extract signature
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, signature_contours, -1, 255, -1)

        # Get bounding box around signature
        x, y, w, h = cv2.boundingRect(np.vstack(signature_contours))
        extracted = img[y:y+h, x:x+w]
        mask_cropped = mask[y:y+h, x:x+w]

        # Make transparent background
        signature_rgba = cv2.cvtColor(extracted, cv2.COLOR_BGR2BGRA)
        signature_rgba[mask_cropped == 0, 3] = 0  # transparent where no signature

        _, buffer = cv2.imencode('.png', signature_rgba)
        b64_output = base64.b64encode(buffer).decode('utf-8')

        if DEBUG_MODE:
            return jsonify({
                "signature_base64": b64_output,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "contour_count": len(signature_contours)
            })

        return jsonify({"signature_base64": b64_output})

    except Exception as e:
        return jsonify({"error": "Processing failed: " + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
