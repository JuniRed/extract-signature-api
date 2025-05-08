from flask import Flask, request, jsonify, send_file
import fitz  # PyMuPDF
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)

def extract_signature_from_pdf(pdf_path):
    # Load the first page of the PDF as an image
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap()
    
    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pix.save(temp_img.name)
    img = cv2.imread(temp_img.name)

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Find contours to detect handwriting-like content
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    signature_img = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 20 and w < 800 and h < 300:  # size filter
            roi = img[y:y+h, x:x+w]
            signature_img = roi
            break

    if signature_img is not None:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        cv2.imwrite(output_path, signature_img)
        return output_path
    else:
        return None

@app.route("/extract-signature", methods=["POST"])
def extract_signature():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        file.save(temp_pdf.name)
        signature_path = extract_signature_from_pdf(temp_pdf.name)

        if signature_path:
            return send_file(signature_path, mimetype="image/png")
        else:
            return jsonify({"error": "Signature not found"}), 404

if __name__ == "__main__":
    app.run()
