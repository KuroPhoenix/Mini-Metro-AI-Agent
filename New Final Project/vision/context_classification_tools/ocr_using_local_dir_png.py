import cv2
import pytesseract

# If Tesseract is not in your PATH, uncomment and edit the next line:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import os

# Optional: Specify tesseract.exe path if it's not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def printRewardForEveryImage():
    for i in range(37):
        # if i != 24: 
        #     continue
        # Set your image filename here (must be in the same directory as this script)
        filename = f"{i+1}.png"

        # Get the absolute path to the file in the same directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_dir = os.path.join(script_dir, 'Weekly Rewards')  # Assuming images are in a subdirectory named 'Weekly Rewards'
        img_path = os.path.join(script_dir, filename)

        # Read the image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Could not open image '{filename}' in {script_dir}")
            exit(1)

        # Define the region (x, y, width, height)
        region =  (419, 490, 209, 68) # left reward text region
        region2 = (676, 496, 210, 63) #

        # Optional: Show the region for debugging
        print(f"@@@Processing image: {filename} in {script_dir}")
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        cv2.imshow("Selected Region", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        x, y, w, h = region2
        roi = image[y:y+h, x:x+w]
        cv2.imshow("Selected Region2", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # Extract and print text
        text = extract_text_from_region(image, region)
        print("Detected text in region:", text)
        text2 = extract_text_from_region(image, region2)
        print("Detected text in region:", text2)



def extract_text_from_region(image, region):
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

    # # (Optional) Convert to grayscale for better OCR accuracy
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # (Optional) Apply thresholding to make text more clear
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Run OCR on the region (use thresh if you thresholded)
    text = pytesseract.image_to_string(thresh)

    return text



if __name__ == "__main__":
    printRewardForEveryImage()
