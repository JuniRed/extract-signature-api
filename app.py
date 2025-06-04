from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os # Added os for DEBUG folder

app = Flask(__name__)

# Added DEBUG and DEBUG_FOLDER from previous version for visualization
DEBUG = True
DEBUG_FOLDER = "debug_images"
if DEBUG and not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)

@app.route('/extract_signature', methods=['POST'])
def extract_signature_api(): # Renamed function to avoid conflict with extract_signature logic
    try:
        data = request.get_json()
        base64_str = data.get("file_base64")
        if not base64_str:
            return jsonify({"error": "No file_base64 provided"}), 400

        # Decode base64 to image
        file_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Call the core extraction function
        signature_img = extract_signature(img)

        # Convert to transparent PNG and encode result
        signature_rgba = cv2.cvtColor(signature_img, cv2.COLOR_BGR2BGRA)
        # Use the mask generated in extract_signature for transparency
        # Assuming extract_signature returns the color image and the mask
        # We need to modify extract_signature to return the mask as well

        # For now, let's re-generate a simple mask or assume the background is white for transparency
        # A more robust approach would be to return the mask from extract_signature
        gray_sig = cv2.cvtColor(signature_rgba, cv2.COLOR_BGR2GRAY)
        _, alpha_mask = cv2.threshold(gray_sig, 240, 255, cv2.THRESH_BINARY_INV)
        signature_rgba[:, :, 3] = alpha_mask


        _, buffer = cv2.imencode('.png', signature_rgba)
        b64_output = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"signature_base64": b64_output})

    except Exception as e:
        # Log the exception for better debugging
        print(f"An error occurred: {e}")
        return jsonify({"error": "Processing failed: " + str(e)}), 500

def extract_signature(img):
    """Extract handwritten signature from image with refined filtering."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "1_gray.png"), gray)

    # Improve contrast and normalize
    gray = cv2.equalizeHist(gray)
    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "1a_equalized.png"), gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "2_blur.png"), blurred)

    # Use Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "3_thresh.png"), thresh)

    # Morphological operations to clean noise and connect ink
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "3a_morph.png"), morph)

    # Find contours on the morphed image
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw accepted contours
    mask = np.zeros_like(thresh)

    # Filter contours based on area, aspect ratio, and standard solidity
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area == 0: # Avoid division by zero for solidity calculation
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h > 0 else 0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        # Calculate standard solidity (area / hull area)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # Refined filtering parameters - these are crucial and might need tuning
        min_area = 100 # Increased minimum area slightly
        max_area_ratio = 0.8 # Max area as a ratio of total image area (prevents large noise blocks)
        min_aspect_ratio = 0.05 # Allow very tall strokes
        max_aspect_ratio = 20.0 # Allow wide strokes, but exclude very long horizontal lines
        min_solidity = 0.1 # Allow relatively irregular shapes
        max_solidity = 0.95 # Exclude very solid shapes (rectangles, thick lines)

        image_area = img.shape[0] * img.shape[1]
        max_area = image_area * max_area_ratio

        # Combined filtering criteria
        if (area > min_area and area < max_area and
            aspect_ratio > min_aspect_ratio and aspect_ratio < max_aspect_ratio and
            solidity > min_solidity and solidity < max_solidity):

            # Draw the contour on the mask if it passes all filters
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "4_filtered_mask.png"), mask)

    # Apply the mask to the original grayscale image
    signature_gray = cv2.bitwise_and(gray, gray, mask=mask)
    if DEBUG:
         cv2.imwrite(os.path.join(DEBUG_FOLDER, "5_signature_gray.png"), signature_gray)


    # Find the combined bounding box of all features in the mask
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Crop the original color image using the combined bounding box
        # Add some padding
        padding = 15 # Increased padding slightly
        y_start = max(0, y - padding)
        y_end = min(img.shape[0], y + h + padding)
        x_start = max(0, x - padding)
        x_end = min(img.shape[1], x + w + padding)

        signature_color = img[y_start:y_end, x_start:x_end]

    else:
        # If no significant contours are found, return a white image
        h, w = img.shape[:2]
        signature_color = np.ones((h, w, 3), dtype=np.uint8) * 255

    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, "6_final_cropped_signature.png"), signature_color)

    # Return the cropped color image. Transparency will be handled in the API route.
    return signature_color


if __name__ == '__main__':
    app.run(debug=True)
