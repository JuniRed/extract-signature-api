from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# Optional: Set debug mode via environment or manual toggle
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

@app.route('/extract_signature', methods=['POST'])
def extract_signature():
    try:
        data = request.get_json()
        base64_str = data.get("file_base64")
        if not base64_str:
            return jsonify({"error": "No file_base64 provided"}), 400

        # Decode base64
        file_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

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

        # First pass filtering (tight)
        signature_contours = filter_contours(contours, tight=True)

        # Fallback: if no signature found, try more relaxed filter
        if not signature_contours:
            signature_contours = filter_contours(contours, tight=False)

        if not signature_contours:
            return jsonify({"error": "No signature-like region found"}), 400

        # Choose the best contour
        c = max(signature_contours, key=lambda c: cv2.contourArea(c))
        x, y, w, h = cv2.boundingRect(c)
        signature_crop = img[y:y+h, x:x+w]

        # Transparent background
        signature_rgba = cv2.cvtColor(signature_crop, cv2.COLOR_BGR2BGRA)
        white_mask = np.all(signature_rgba[:, :, :3] > 240, axis=-1)
        signature_rgba[white_mask, 3] = 0

        # Encode result
        _, buffer = cv2.imencode('.png', signature_rgba)
        b64_output = base64.b64encode(buffer).decode('utf-8')

        # Optional debug return
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

        # Tight filter (for clean input)
        if tight:
            if area > 1000 and 0.5 < aspect_ratio < 10 and solidity > 0.15:
                result.append(cnt)
        else:
            # Loose filter (for faint or broken signatures)
            if area > 500 and 0.2 < aspect_ratio < 15 and solidity > 0.08:
                result.append(cnt)
    return result


if __name__ == '__main__':
    app.run(debug=True)
