from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import numpy as np
import cv2
import fitz  # PyMuPDF

app = Flask(__name__)

def extract_image_from_pdf(file_bytes):
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    if len(pdf) == 0:
        raise ValueError("Empty PDF")
    page = pdf[0]
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")
    return img_bytes

def extract_signature_from_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found")

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    signature_crop = image_cv[y:y+h, x:x+w]

    # Create transparent image
    cropped_rgb = cv2.cvtColor(signature_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cropped_rgb)
    pil_img = pil_img.convert("RGBA")
    datas = pil_img.getdata()
    newData = []
    for item in datas:
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            newData.append((255, 255, 255, 0))  # Transparent
        else:
            newData.append(item)
    pil_img.putdata(newData)

    output_buffer = io.BytesIO()
    pil_img.save(output_buffer, format='PNG')
    return base64.b64encode(output_buffer.getvalue()).decode('utf-8')

@app.route('/extract_signature', methods=['POST'])
def extract_signature():
    try:
        data = request.get_json()
        file_base64 = data.get("file_base64", "")
        if not file_base64:
            return jsonify({"error": "Missing file_base64"}), 400

        file_bytes = base64.b64decode(file_base64)

        # Try PDF first, fallback to image
        try:
            image_bytes = extract_image_from_pdf(file_bytes)
        except Exception:
            image_bytes = file_bytes  # assume it's an image

        signature_base64 = extract_signature_from_image(image_bytes)
        return jsonify({"signature_base64": signature_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
