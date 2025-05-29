import cv2

# 1. Load your frame
img = cv2.imread('HK.png')

# 2. Draw a box around the passenger count; press Enter or Space when you’re happy
x, y, w, h = cv2.selectROI('Determine_ROI', img, showCrosshair=True)

cv2.destroyAllWindows()
print(f'ROI: x={x}, y={y}, w={w}, h={h}')
