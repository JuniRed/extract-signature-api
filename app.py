from flask import Flask, request, jsonify
from PIL import Image
import cv2
import numpy as np
import fitz  # PyMuPDF
import io
import base64
import requests
from datetime import datetime

app = Flask(__name__)

def convert_pdf_to_image(file_stream):
    doc = fitz.open(stream=file_stream, filetype="pdf")
    page = doc.load_page(0)  # first page
    pix = page.get_pixmap()
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    return image

def extract_signature(image: Image.Image) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    signature_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(signature_contour)
    signature_crop = image_np[y:y+h, x:x+w]

    signature_image = Image.fromarray(signature_crop)
    return signature_image

def upload_to_sharepoint(image: Image.Image, filename: str):
    sharepoint_url = "<YOUR_SHAREPOINT_UPLOAD_URL>"  # Replace this
    access_token = "<YOUR_ACCESS_TOKEN>"  # Get via Azure AD

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json;odata=verbose",
        "Content-Type": "image/png"
    }
    response = requests.post(
        sharepoint_url,
        headers=headers,
        data=buffered.getvalue()
    )
    return response.status_code, response.text

@app.route("/extract-signature", methods=["POST"])
def extract_signature_api():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = file.filename.lower()
    image = None

    try:
        if filename.endswith(".pdf"):
            image = convert_pdf_to_image(file.stream)
        else:
            image = Image.open(file.stream)

        signature = extract_signature(image)

        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        image_filename = f"signature_{now}.png"

        status_code, msg = upload_to_sharepoint(signature, image_filename)

        if status_code == 200 or status_code == 201:
            return jsonify({"message": "Signature extracted and uploaded successfully", "filename": image_filename}), 200
        else:
            return jsonify({"error": "Failed to upload to SharePoint", "details": msg}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
