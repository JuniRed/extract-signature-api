from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import fitz  # PyMuPDF
import io
import base64

app = Flask(__name__)

def convert_pdf_to_image(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap()
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    return image

def extract_signature(image: Image.Image) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Exception("No signature found")

    signature_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(signature_contour)
    signature_crop = image_np[y:y+h, x:x+w]
    return Image.fromarray(signature_crop)

@app.route("/extract-signature", methods=["POST"])
def extract_signature_api():
    data = request.get_json()
    if not data or "file_base64" not in data:
        return jsonify({"error": "Missing 'file_base64' in request"}), 400

    file_b64 = data["file_base64"]
    
    # Remove data URL prefix if present
    if "," in file_b64:
        file_b64 = file_b64.split(",", 1)[1]

    try:
        file_bytes = base64.b64decode(file_b64)
        is_pdf = file_bytes[:4] == b'%PDF'

        image = convert_pdf_to_image(file_bytes) if is_pdf else Image.open(io.BytesIO(file_bytes))

        signature_image = extract_signature(image)
        buffer = io.BytesIO()
        signature_image.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return jsonify({"signature_base64": encoded_image}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
