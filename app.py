from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import numpy as np
import cv2
import base64
import os
import io

app = Flask(__name__)

DEBUG = True
DEBUG_FOLDER = "debug_images"
if DEBUG and not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)

def pdf_to_image(pdf_bytes):
    """Convert the first page of a PDF to a high-resolution image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        raise ValueError("PDF has no pages.")
    page = doc.load_page(0)
    zoom = 4  # 300 DPI equivalent
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def bytes_to_image(image_bytes):
    """Convert image bytes to a NumPy array."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes.")
    return img

def extract_signature(img):
    """Extract handwritten signature from image with refined filtering."""
    if img is None or img.size == 0:
         raise ValueError("Input image is empty.")

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
        # Ensure image_area is not zero before division
        max_area = image_area * max_area_ratio if image_area > 0 else float('inf')


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

        # Ensure valid slicing indices
        if y_start >= y_end or x_start >= x_end:
             signature_color = np.ones((50, 150, 3), dtype=np.uint8) * 255 # Return a small white image if crop is invalid
        else:
             signature_color = img[y_start:y_end, x_start:x_end]

    else:
        # If no significant contours are found, return a white image
        h, w = img.shape[:2]
        # Ensure dimensions are valid before creating white image
        if h == 0 or w == 0:
             signature_color = np.ones((50, 150, 3), dtype=np.uint8) * 255 # Return a small white image
        else:
            signature_color = np.ones((h, w, 3), dtype=np.uint8) * 255

    if DEBUG:
        # Check if signature_color is empty before saving
        if signature_color is not None and signature_color.size > 0:
             cv2.imwrite(os.path.join(DEBUG_FOLDER, "6_final_cropped_signature.png"), signature_color)
        else:
             print("Debug: signature_color is empty, skipping save of 6_final_cropped_signature.png")


    # Return the cropped color image and the mask for potential transparency handling outside
    return signature_color, mask


@app.route('/extract_signature', methods=['POST'])
def extract_signature_api():
    try:
        data = request.get_json()
        pdf_base64 = data.get("pdf_base64")
        image_base64 = data.get("image_base64")

        img = None
        if pdf_base64:
            pdf_bytes = base64.b64decode(pdf_base64)
            img = pdf_to_image(pdf_bytes)
        elif image_base64:
            image_bytes = base64.b64decode(image_base64)
            img = bytes_to_image(image_bytes)
        else:
            return jsonify({"error": "No pdf_base64 or image_base64 provided"}), 400

        # Call the core extraction function, which now returns the cropped image and the mask
        signature_img_color, signature_mask = extract_signature(img)

        # Convert the extracted color image to RGBA for transparency
        signature_rgba = cv2.cvtColor(signature_img_color, cv2.COLOR_BGR2BGRA)

        # Resize the mask to match the cropped signature image dimensions
        # This is necessary because the mask is generated on the original image dimensions
        # and the signature_img_color is a cropped version.
        # Find the bounding box of the mask on the original image to get the crop region
        coords = cv2.findNonZero(signature_mask)
        if coords is not None:
             x, y, w, h = cv2.boundingRect(coords)
             # Apply the same padding as used in extract_signature
             padding = 15
             y_start = max(0, y - padding)
             y_end = min(signature_mask.shape[0], y + h + padding)
             x_start = max(0, x - padding)
             x_end = min(signature_mask.shape[1], x + w + padding)

             # Ensure valid slicing indices
             if y_start < y_end and x_start < x_end:
                 cropped_mask = signature_mask[y_start:y_end, x_start:x_end]
                 # Ensure the cropped mask has the same dimensions as the cropped color image
                 if cropped_mask.shape[:2] == signature_rgba.shape[:2]:
                     # Set the alpha channel using the cropped mask
                     signature_rgba[:, :, 3] = cropped_mask
                 else:
                    # If dimensions don't match, fall back to simple white background transparency
                    print("Debug: Cropped mask dimensions do not match cropped image. Falling back to simple transparency.")
                    gray_sig = cv2.cvtColor(signature_rgba, cv2.COLOR_BGR2GRAY)
                    _, alpha_mask = cv2.threshold(gray_sig, 240, 255, cv2.THRESH_BINARY_INV)
                    signature_rgba[:, :, 3] = alpha_mask

             else:
                 # If cropping coordinates are invalid, fall back to simple white background transparency
                 print("Debug: Invalid cropping coordinates for mask. Falling back to simple transparency.")
                 gray_sig = cv2.cvtColor(signature_rgba, cv2.COLOR_BGR2GRAY)
                 _, alpha_mask = cv2.threshold(gray_sig, 240, 255, cv2.THRESH_BINARY_INV)
                 signature_rgba[:, :, 3] = alpha_mask

        else:
            # If no coordinates in mask, fall back to simple white background transparency
            print("Debug: No coordinates found in mask. Falling back to simple transparency.")
            gray_sig = cv2.cvtColor(signature_rgba, cv2.COLOR_BGR2GRAY)
            _, alpha_mask = cv2.threshold(gray_sig, 240, 255, cv2.THRESH_BINARY_INV)
            signature_rgba[:, :, 3] = alpha_mask


        # Encode result
        _, buffer = cv2.imencode('.png', signature_rgba)
        b64_output = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"signature_base64": b64_output})

    except Exception as e:
        # Log the exception for better debugging
        print(f"An error occurred: {e}")
        return jsonify({"error": "Processing failed: " + str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
