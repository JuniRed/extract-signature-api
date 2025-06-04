from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import numpy as np
import cv2
import base64
import os

app = Flask(__name__)

# Optional debug folder to save intermediate images
DEBUG = True
DEBUG_FOLDER = "debug_images"
if DEBUG and not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)

def pdf_to_image(pdf_bytes):
    """Convert first page of PDF bytes to a high-res RGB image (numpy array)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    zoom = 4  # high resolution
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def extract_signature(img):
    """Extract signature using enhanced processing with debug saves."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "1_gray.png"), gray)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "2_blur.png"), blur)
    
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "3_thresh.png"), thresh)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        h, w = img.shape[:2]
        return np.ones((h, w, 3), dtype=np.uint8) * 255

    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < 500:
        h, w = img.shape[:2]
        return np.ones((h, w, 3), dtype=np.uint8) * 255

    x, y, w, h = cv2.boundingRect(largest_contour)

    roi = thresh[y:y+h, x:x+w]

    signature = cv2.bitwise_not(roi)
    
    signature_color = cv2.cvtColor(signature, cv2.COLOR_GRAY2BGR)

    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "4_signature.png"), signature_color)

    return signature_color

def image_to_base64(img):
    """Encode OpenCV image as base64 PNG string."""
    _, buffer = cv2.imencode('.png', img)
    img_bytes = buffer.tobytes()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str

@app.route('/extract-signature', methods=['POST'])
def extract_signature_api():
    try:
        if 'file' in request.files:
            pdf_file = request.files['file']
            pdf_bytes = pdf_file.read()
        else:
            data = request.get_json(force=True)
            pdf_base64 = data.get('pdf_base64', '')
            if not pdf_base64:
                return jsonify({"error": "No PDF provided"}), 400
            pdf_bytes = base64.b64decode(pdf_base64)
        
        img = pdf_to_image(pdf_bytes)
        signature_img = extract_signature(img)
        signature_b64 = image_to_base64(signature_img)
        
        return jsonify({"signature_base64": signature_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
