import cv2
import numpy as np

# 1. load and blur to remove noise
img = cv2.imread('/frame_1747706474.5595012.png')
blur = cv2.GaussianBlur(img, (5,5), 0)

# 2. convert to HSV and threshold for bright (white) regions
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
lower = np.array([  0,   0, 200])   # very low saturation, high value
upper = np.array([180,  30, 255])
mask = cv2.inRange(hsv, lower, upper)

# 3. clean up small speckles
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 4. find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5. filter for triangles
triangles = []
for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    if len(approx) == 3 and cv2.contourArea(cnt) > 50:  # area>50 to skip tiny noise
        triangles.append(cnt)

# 6. build a mask of just the triangles
tri_mask = np.zeros_like(mask)
cv2.drawContours(tri_mask, triangles, -1, 255, thickness=cv2.FILLED)

# 7. extract the colored triangle-patches
result = cv2.bitwise_and(img, img, mask=tri_mask)
cv2.imwrite('triangles.png', result)
