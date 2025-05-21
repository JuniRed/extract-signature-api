from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io
import base64
from pdf2image import convert_from_bytes

app = Flask(__name__)

def extract_signature(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    signature = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h < 150:  # Filter small text, big blobs
            cropped = image_np[y:y+h, x:x+w]
            signature = Image.fromarray(cropped)
            break

    return signature

@app.route('/extract_signature', methods=['POST'])
def extract_signature_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = file.filename.lower()

    try:
        if filename.endswith('.pdf'):
            pages = convert_from_bytes(file.read())
            image = np.array(pages[0])  # First page only
        else:
            img_bytes = file.read()
            np_arr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sig_img = extract_signature(image)
        if sig_img:
            buffer = io.BytesIO()
            sig_img.save(buffer, format='PNG')
            base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return jsonify({'signature_image': base64_img})
        else:
            return jsonify({'error': 'Signature not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
