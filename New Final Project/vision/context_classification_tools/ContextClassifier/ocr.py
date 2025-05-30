import cv2
import pytesseract
import os
import numpy as np
import re
from PIL import Image
import numpy as np
# (x, y, w, h) of the passenger counter – adjust once and reuse
COUNTER_ROI = (1063, 57, 56, 21)

def read_screenshot() -> np.ndarray:
    """
    Grab one frame with pyautogui (remove in tests and pass in your own img).
    """
    import pyautogui
    x, y, w, h = 0, 25, 1280, 828
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

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
    # cv2.imshow("Selected roi", roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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

def _preprocess_counter(img: np.ndarray) -> np.ndarray:
    """
    Enhance contrast so that Tesseract sees sharp white digits on black.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # adaptive threshold helps if background brightness drifts
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 8
    )

    # Optional: enlarge characters, remove small specks
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    return thresh


def passenger_cnt(frame: np.ndarray | None = None) -> int:
    """
    Return the number shown in the in-game passenger counter.

    Parameters
    ----------
    frame : BGR image (full screenshot).  If None, grabs one.

    Returns
    -------
    int – parsed passenger count, or 0 if OCR fails.
    """
    if frame is None:
        frame = read_screenshot()
    # if someone passed us a PIL Image, turn it into an ndarray
    if isinstance(frame, Image.Image):
         frame = np.array(frame)
    x, y, w, h = COUNTER_ROI
    roi = frame[y:y + h, x:x + w]

    processed = _preprocess_counter(roi)

    custom_cfg = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(processed, config=custom_cfg)

    # keep only digits, fallback to 0
    digits = re.sub(r'\D', '', text)
    return int(digits) if digits else 0
