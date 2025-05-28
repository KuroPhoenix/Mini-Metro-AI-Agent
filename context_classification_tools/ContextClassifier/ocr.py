import cv2
import pytesseract
import os
import numpy as np


def extract_text_from_region(image: np.ndarray, region: tuple): #image: ndarray
    """
    Extracts text from a specific region in an image using Tesseract OCR.

    Args:
        image: The input image (numpy array, as read by cv2).
        region: A tuple (x, y, w, h) specifying the top-left corner and size of the region.

    Returns:
        Detected text string.
    """
    x, y, w, h = region

    # Crop the image to the region of interest
    roi = image[y:y+h, x:x+w]
    print("region tuple:", region)
    print("raw roi.shape:", roi.shape)

    ###
    cv2.imshow("Selected roi", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # (Optional) Convert to grayscale for better OCR accuracy
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # (Optional) Apply thresholding to make text more clear
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Run OCR on the region (use thresh if you thresholded)
    text = pytesseract.image_to_string(thresh)

    return text

def read_png():
        filename = "context.png"

        # Get the absolute path to the file in the same directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(script_dir, filename)

        # Read the image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Could not open image '{filename}' in {script_dir}")
            exit(1)

        return image


def ocr_in_region(region):
    image = read_png()

    # Extract and print text
    text = extract_text_from_region(image, region)

    return text
