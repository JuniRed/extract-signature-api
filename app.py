from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import numpy as np
import cv2
import base64
import io
from PIL import Image

app = Flask(__name__)

def pdf_to_image(pdf_bytes):
    """Convert first page of PDF bytes to RGB image (numpy array)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)  # first page
    zoom = 3  # zoom factor for better resolution
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:  # RGBA to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def extract_signature(img):
    """
    Extract handwritten signature from image by:
    - converting to grayscale,
    - adaptive thresholding,
    - morphological operations to isolate dark ink,
    - contour detection to find the signature,
    - cropping and cleaning background.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert image so signature is white on black background
    inv = cv2.bitwise_not(gray)
    
    # Adaptive threshold to isolate signature strokes
    thresh = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 25, 15)
    
    # Morphological closing to join broken parts of the signature
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours and filter by area (remove noise)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours, return blank white image
    if not contours:
        h, w = img.shape[:2]
        return np.ones((h, w), dtype=np.uint8) * 255
    
    # Find bounding box around all contours combined
    x_min = min([cv2.boundingRect(c)[0] for c in contours])
    y_min = min([cv2.boundingRect(c)[1] for c in contours])
    x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours])
    y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours])
    
    # Crop to bounding box
    signature_roi = closed[y_min:y_max, x_min:x_max]
    
    # Create white background image
    bg = np.ones_like(signature_roi) * 255
    
    # Invert signature ROI to black strokes on white
    signature_img = cv2.bitwise_not(signature_roi)
    
    # Convert to 3 channel BGR image for saving/display
    signature_img_color = cv2.cvtColor(signature_img, cv2.COLOR_GRAY2BGR)
    
    return signature_img_color

def image_to_base64(img):
    """Convert OpenCV image (BGR) to base64 PNG string."""
    _, buffer = cv2.imencode('.png', img)
    img_bytes = buffer.tobytes()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str

@app.route('/extract-signature', methods=['POST'])
def extract_signature_api():
    """
    POST endpoint to receive PDF and return extracted signature.
    
    Accepts multipart/form-data with key 'file' (PDF),
    or JSON with 'pdf_base64' key containing base64 PDF string.
    
    Returns JSON: { "signature_base64": "<base64 PNG>" }
    """
    try:
        if 'file' in request.files:
            pdf_file = request.files['file']
            pdf_bytes = pdf_file.read()
        else:
            data = request.get_json(force=True)
            pdf_base64 = data.get('pdf_base64', '')
            if not pdf_base64:
                return jsonify({"error": "No PDF file or base64 provided"}), 400
            pdf_bytes = base64.b64decode(pdf_base64)
        
        # Convert PDF to image
        img = pdf_to_image(pdf_bytes)
        
        # Extract signature image
        signature_img = extract_signature(img)
        
        # Encode extracted signature as base64 PNG
        signature_b64 = image_to_base64(signature_img)
        
        return jsonify({"signature_base64": signature_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
