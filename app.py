from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
os.makedirs("signatures", exist_ok=True)

def convert_pdf_to_image(pdf_base64):
    pdf_bytes = base64.b64decode(pdf_base64)
    doc = fitz.open("pdf", pdf_bytes)
    page = doc[0]
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def detect_signature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 2 < w/h < 10 and 50 < w < 400:
            return image[y:y+h, x:x+w]
    return None

@app.route("/extract-signature", methods=["POST"])
def extract_signature():
    data = request.get_json()
    file_type = data["file_type"]  # "pdf" or "image"
    base64_data = data["file_data"]

    if file_type == "pdf":
        image = convert_pdf_to_image(base64_data)
    else:
        image_bytes = base64.b64decode(base64_data)
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    signature = detect_signature(image)
    if signature is not None:
        filename = "signatures/signature.png"
        cv2.imwrite(filename, signature)
        return jsonify({"success": True, "message": "Signature saved", "file": filename})
    else:
        return jsonify({"success": False, "message": "Signature not found"})

if __name__ == "__main__":
    app.run(debug=True)
