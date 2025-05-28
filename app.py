from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

def extract_signature(image_np):
    """Basic signature extraction using OpenCV."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest contour (assumed signature)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        signature = image_np[y:y+h, x:x+w]
        return signature
    return None

@app.route('/extract_signature', methods=['POST'])
def extract_signature_api():
    try:
        # Handle base64 image
        if 'image_base64' in request.json:
            img_data = request.json['image_base64']
            img_bytes = base64.b64decode(img_data.split(',')[-1])
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
            image_np = np.array(image)

        # Handle file upload
        elif 'file' in request.files:
            file = request.files['file']
            img_bytes = file.read()
            file.stream.seek(0)  # Reset the file pointer in case it's needed again
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
            image_np = np.array(image)

        else:
            return jsonify({'error': 'No image data provided'}), 400

        # Extract signature
        signature = extract_signature(image_np)
        if signature is None:
            return jsonify({'error': 'No signature found'}), 404

        # Encode extracted signature to base64
        _, buffer = cv2.imencode('.png', signature)
        signature_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'signature_image': f'data:image/png;base64,{signature_base64}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
