from flask import Flask, request, jsonify, send_file, make_response
import cv2
import numpy as np
import base64
import io

app = Flask(__name__)

def decode_base64_image(img_b64):
    """Decode a base64-encoded image to a BGR NumPy array."""
    try:
        img_data = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        raise ValueError("Invalid image data") from e

def preprocess_image(image):
    """Convert to grayscale, denoise, and binarize to isolate dark strokes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu's thresholding (invert so signature is white on black background):contentReference[oaicite:9]{index=9}.
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    # Morphological opening (remove small noise) and closing (bridge signature strokes):contentReference[oaicite:10]{index=10}.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed

def find_signature_contours(binary, image_shape):
    """Find and filter contours likely to be the signature using heuristics."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = image_shape[:2]
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or area > (img_w * img_h * 0.9):
            # Skip tiny noise or huge background
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / max(h, 1)
        if aspect_ratio < 2.0:
            # Signature is usually wider than tall:contentReference[oaicite:11]{index=11}
            continue
        extent = area / float(w * h)
        if extent > 0.6:
            # Likely not scribbly enough (too solid):contentReference[oaicite:12]{index=12}
            continue
        # Compute solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = float(area) / hull_area
        if solidity > 0.8:
            # Too convex (e.g. stamp or solid shape):contentReference[oaicite:13]{index=13}
            continue
        # Check vertical position: ignore contours in top 30% of image
        if y < img_h * 0.3:
            continue
        candidates.append(cnt)
    return candidates

def build_signature_mask(contours, shape):
    """Create a mask image (255 in signature region, 0 elsewhere) from contours."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if not contours:
        return mask
    # Fill all candidate contours
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)
    return mask

def create_transparent_signature(image, mask):
    """
    Create a BGRA image where pixels under the mask are kept (with alpha=255)
    and all other pixels are transparent.
    """
    b, g, r = cv2.split(image)
    # Mask the color channels
    b = cv2.bitwise_and(b, mask)
    g = cv2.bitwise_and(g, mask)
    r = cv2.bitwise_and(r, mask)
    # Use the mask itself as alpha channel (255 inside signature, 0 outside)
    rgba = cv2.merge((b, g, r, mask))
    return rgba

@app.route('/extract_signature', methods=['POST'])
def extract_signature():
    try:
        data = request.get_json(force=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'Missing base64 image data'}), 400

        image = decode_base64_image(data['image'])
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        # Optionally resize large images for speed (e.g., max width=1200):contentReference[oaicite:14]{index=14}.
        max_width = 1200
        h, w = image.shape[:2]
        if w > max_width:
            scale = max_width / w
            image = cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

        # Preprocess to get a binary image of potential signatures
        binary = preprocess_image(image)

        # Find contours that may correspond to the signature
        signature_contours = find_signature_contours(binary, image.shape)
        if not signature_contours:
            return jsonify({'error': 'Signature not found'}), 404

        # Build mask and extract signature
        mask = build_signature_mask(signature_contours, image.shape)
        signature_rgba = create_transparent_signature(image, mask)

        # Encode RGBA image to PNG in memory
        success, png = cv2.imencode('.png', signature_rgba)
        if not success:
            raise RuntimeError("Failed to encode PNG")

        # Return as PNG image response
        buf = io.BytesIO(png.tobytes())
        response = make_response(buf.getvalue())
        response.headers.set('Content-Type', 'image/png')
        return response

    except Exception as e:
        # Log the error in real applications
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
