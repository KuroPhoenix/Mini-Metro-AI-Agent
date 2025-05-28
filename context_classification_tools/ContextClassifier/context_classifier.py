import pyautogui
import cv2
import numpy as np
import time

import ocr

cnt = 1
while True:
    
    # region_whole = (0, 30, 638, 472)  # (x1, y1, x2, y2)
    frame = pyautogui.screenshot()  # pyautogui.screenshot(region=(left, top, width, height))
    # img = np.array(frame.convert("RGB"))
    # #img   = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # print("raw screenshot shape:", img.shape, "min/max:", img.min(), img.max())
    # cv2.imwrite("debug_full.png", img)
    frame.save("./ContextClassifier/context.png") # expect you run the py file under outer directory

    time.sleep(0.5)


    # Use your own region
    region_left = (402, 610, 647-402, 50) # (x1, y1, w, h)
    #"REGION": (610, 419, 668, 637),   (y1, x1, y2, x2) from spritex copy
    region_right = (662, 614, 899-662, 50)
   
    # x, y, w, h = region_left
    # i = img[y:y+h, x:x+w]
    # cv2.imshow("hi_test", i)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    text_left = ocr.ocr_in_region(region=region_left)
    text_right = ocr.ocr_in_region(region=region_right)
    
    print(f"{cnt}: Left: {text_left}\nRight: {text_right}")
    cnt += 1

    # logic to determine(can use what you prefer, e.g. a dict store all reward names)
    print(len(text_left), len(text_right))
    print("left:")
    for _ in text_left:
        print(f"*{_}*")

    if text_left == "" and text_right == "" :
        print(f"{cnt}: In gameplay context")
    elif "Loco" in text_left and "tive" in text_right and len(text_left) <= 6 and len(text_right) <= 6:
        print(f"{cnt}: In 1-reward context")
    else:
        print(f"{cnt}: In 2-reward context")
