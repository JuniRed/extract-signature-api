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

        # Decode as image or PDF
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            try:
                pil_images = convert_from_bytes(file_bytes)
                img_pil = pil_images[0]
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                return jsonify({"error": "Invalid image/PDF: " + str(e)}), 400

        # Step 1: Grayscale + Blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Step 2: Adaptive threshold to detect strokes (inverted)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )

        # Step 3: Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Step 4: Find contours
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        total_area = w * h

        signature_mask = np.zeros_like(gray)

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = cw / float(ch) if ch > 0 else 0
            arc_len = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * arc_len, True)

            # Signature-like features
            if (
                500 < area < 0.03 * total_area and
                2 < aspect_ratio < 10 and
                len(approx) > 15
            ):
                cv2.drawContours(signature_mask, [cnt], -1, 255, -1)

        if np.count_nonzero(signature_mask) == 0:
            return jsonify({"error": "No signature detected"}), 400

        # Step 5: Crop to signature
        ys, xs = np.where(signature_mask == 255)
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        cropped_img = img[y_min:y_max, x_min:x_max]
        cropped_mask = signature_mask[y_min:y_max, x_min:x_max]

        # Step 6: Apply mask to make background transparent
        signature_rgba = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)
        signature_rgba[cropped_mask == 0, 3] = 0  # Transparent where not signature

        _, buffer = cv2.imencode('.png', signature_rgba)
        b64_output = base64.b64encode(buffer).decode('utf-8')

        if DEBUG_MODE:
            return jsonify({
                "signature_base64": b64_output,
                "bbox": {"x": int(x_min), "y": int(y_min), "w": int(x_max - x_min), "h": int(y_max - y_min)},
                "signature_pixels": int(np.count_nonzero(signature_mask))
            })

        return jsonify({"signature_base64": b64_output})

    except Exception as e:
        return jsonify({"error": "Processing failed: " + str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
