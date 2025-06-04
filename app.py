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
    """Extract handwritten signature from image with noise filtering and morphological operations."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "1_gray.png"), gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "2_blur.png"), blurred)

    # Use Otsu's thresholding for better automatic threshold determination
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "3_thresh.png"), thresh)

    # Add morphological operations to connect signature strokes and remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morphed_thresh = cv2.morphologyEx(morphed_thresh, cv2.MORPH_OPEN, kernel)
    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "3a_morphed_thresh.png"), morphed_thresh)

    contours, _ = cv2.findContours(morphed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(thresh)

    # Filter contours based on area, aspect ratio, and solidity
    for contour in contours:
        area = cv2.contourArea(contour)
        if area == 0: # Avoid division by zero
            continue

        x, y, w, h = cv2.boundingRect(contour)
        # Calculate aspect ratio (width / height)
        aspect_ratio = w / float(h) if h != 0 else 0

        # Calculate solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # Adjust filtering parameters based on typical signature characteristics
        # These values might need further tuning based on your specific images
        min_area = 50
        max_area = 50000
        min_aspect_ratio = 0.1 # Allow relatively tall and thin strokes
        max_aspect_ratio = 10.0 # Allow relatively wide and short strokes
        min_solidity = 0.3 # Exclude very irregular shapes (might be noise)
        max_solidity = 0.9 # Exclude very solid shapes (like rectangles or lines)


        if (min_area < area < max_area and
            min_aspect_ratio < aspect_ratio < max_aspect_ratio and
            min_solidity < solidity < max_solidity):
             cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)


    signature = cv2.bitwise_and(gray, gray, mask=mask)

    # Invert the signature colors to have dark signature on white background
    signature_inv = cv2.bitwise_not(signature)
    signature_color = cv2.cvtColor(signature_inv, cv2.COLOR_GRAY2BGR)

    # Find the bounding box of the extracted signature based on the mask
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Crop the signature image to the bounding box
        # Add some padding around the bounding box to ensure no parts are cut off
        padding = 10
        y_start = max(0, y - padding)
        y_end = min(img.shape[0], y + h + padding)
        x_start = max(0, x - padding)
        x_end = min(img.shape[1], x + w + padding)
        signature_color = signature_color[y_start:y_end, x_start:x_end]
    else:
        # If no signature is found, return a white image
        h, w = img.shape[:2]
        signature_color = np.ones((h, w, 3), dtype=np.uint8) * 255


    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "4_final_signature.png"), signature_color)

    return signature_color

def image_to_base64(img):
    """Encode image as base64 string."""
    # Ensure the image is not empty before encoding
    if img is None or img.size == 0:
        return ""
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
            elif filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')): # Added more image types
                img = bytes_to_image(file_bytes)
            else:
                return jsonify({"error": "Unsupported file type. Please provide PDF, PNG, JPG, JPEG, BMP, or TIFF."}), 400

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
        # Log the exception for better debugging
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
