from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import numpy as np
import cv2
import base64
import os
import io

app = Flask(__name__)

DEBUG = True
DEBUG_FOLDER = "debug_images"
if DEBUG and not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)

def pdf_to_image(pdf_bytes):
    """Convert the first page of a PDF to a high-resolution image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    zoom = 4  # 300 DPI equivalent
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def bytes_to_image(image_bytes):
    """Convert image bytes to a NumPy array."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes.")
    return img

def extract_signature(img):
    """Extract handwritten signature from image with noise filtering."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "1_gray.png"), gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "2_blur.png"), blurred)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8
    )
    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "3_thresh.png"), thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(thresh)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 1000 < area < 50000:  # Adjust range as needed
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    signature = cv2.bitwise_and(gray, gray, mask=mask)

    signature_inv = cv2.bitwise_not(signature)
    signature_color = cv2.cvtColor(signature_inv, cv2.COLOR_GRAY2BGR)

    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        signature_color = signature_color[y:y+h, x:x+w]
    else:
        h, w = img.shape[:2]
        signature_color = np.ones((h, w, 3), dtype=np.uint8) * 255

    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "4_final_signature.png"), signature_color)

    return signature_color

def image_to_base64(img):
    """Encode image as base64 string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/extract-signature', methods=['POST'])
def extract_signature_api():
    try:
        img = None
        if 'file' in request.files:
            file = request.files['file']
            filename = file.filename.lower()
            file_bytes = file.read()
            if filename.endswith('.pdf'):
                img = pdf_to_image(file_bytes)
            elif filename.endswith(('.png', '.jpg', '.jpeg')):
                img = bytes_to_image(file_bytes)
            else:
                return jsonify({"error": "Unsupported file type. Please provide PDF, PNG, JPG, or JPEG."}), 400

        elif request.get_json():
            data = request.get_json(force=True)
            pdf_base64 = data.get('pdf_base64', '')
            image_base64 = data.get('image_base64', '')

            if pdf_base64:
                pdf_bytes = base64.b64decode(pdf_base64)
                img = pdf_to_image(pdf_bytes)
            elif image_base64:
                image_bytes = base64.b64decode(image_base64)
                img = bytes_to_image(image_bytes)
            else:
                return jsonify({"error": "No file or base64 data provided (expected 'file', 'pdf_base64', or 'image_base64')."}), 400
        else:
             return jsonify({"error": "No file or data provided."}), 400

        if img is None:
             return jsonify({"error": "Could not process input file or data."}), 400


        signature_img = extract_signature(img)
        signature_b64 = image_to_base64(signature_img)

        return jsonify({"signature_base64": signature_b64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
