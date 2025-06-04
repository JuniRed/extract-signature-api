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

        # Try decoding as image
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        pages = []

        # If image decode fails, try PDF
        if img is None:
            try:
                pil_images = convert_from_bytes(file_bytes)
                for page in pil_images:
                    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                    pages.append(img)
            except Exception as e:
                return jsonify({"error": "Could not decode as image or PDF: " + str(e)}), 400
        else:
            pages.append(img)

        for page_num, img in enumerate(pages):
            # Preprocess
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Binary inverse for handwriting
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours using enhanced logic
            signature_contours = filter_contours(contours, img.shape)

            if not signature_contours:
                continue  # Try next page

            # Merge nearby contours
            signature_contours = merge_close_contours(signature_contours)

            # Choose largest
            c = max(signature_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            signature_crop = img[y:y + h, x:x + w]

            # Transparent background
            signature_rgba = cv2.cvtColor(signature_crop, cv2.COLOR_BGR2BGRA)
            white_mask = np.all(signature_rgba[:, :, :3] > 240, axis=-1)
            signature_rgba[white_mask, 3] = 0

            _, buffer = cv2.imencode('.png', signature_rgba)
            b64_output = base64.b64encode(buffer).decode('utf-8')

            response = {
                "signature_base64": b64_output,
                "page": page_num + 1,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "contour_count": len(contours),
                "selected_contours": len(signature_contours)
            }

            return jsonify(response)

        return jsonify({"error": "No signature-like region found on any page"}), 400

    except Exception as e:
        return jsonify({"error": "Processing failed: " + str(e)}), 500


def filter_contours(contours, image_shape):
    result = []
    img_h, img_w = image_shape[:2]
    img_area = img_w * img_h

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = w / float(h) if h > 0 else 0
        solidity = area / float(w * h + 1e-5)
        density = area / (w + h + 1e-5)

        if area < 0.001 * img_area or area > 0.1 * img_area:
            continue

        if (
            2 < aspect_ratio < 15 and      # Long and narrow
            0.1 < solidity < 0.5 and       # Hollow-ish = handwriting
            density > 5                    # Dense lines
        ):
            result.append(cnt)
    return result


def merge_close_contours(contours, max_distance=50):
    merged = []
    while contours:
        base = contours.pop(0)
        base_box = cv2.boundingRect(base)
        bx, by, bw, bh = base_box
        merged_cnt = base.copy()

        to_merge = []
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if abs(x - bx) < max_distance and abs(y - by) < max_distance:
                merged_cnt = np.concatenate((merged_cnt, cnt), axis=0)
                to_merge.append(i)

        for i in sorted(to_merge, reverse=True):
            contours.pop(i)
        merged.append(merged_cnt)
    return merged


if __name__ == '__main__':
    app.run(debug=True)
