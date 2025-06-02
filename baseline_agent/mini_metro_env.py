import numpy as np
import random
import cv2
import pyautogui
import time
import math
from collections import defaultdict

class MiniMetroEnv:
    def __init__(self):
        self.known_stations = {'circle': set(), 'triangle': set(), 'square': set()}
        self.connected_pairs = set()

    def screenshot_game_area(self):
        x, y, width, height = 90, 85, 1280, 800
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        return screenshot, x, y

    def detect_weekly_reward_screen(self):
        screenshot, offset_x, offset_y = self.screenshot_game_area()
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('reward_template.png', 0)
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)

        if len(loc[0]) > 0:
            # 找到匹配，返回第一個匹配的位置
            pt = (loc[1][0] + offset_x, loc[0][0] + offset_y)
            return pt
        else:
            return None

    def click_weekly_reward(self):
        pt = self.detect_weekly_reward_screen()
        if pt:
            # 模擬點擊獎勵選項
            pyautogui.moveTo(pt[0], pt[1], duration=0.3)
            pyautogui.click()
            print("已選擇每週獎勵。")
            time.sleep(1)  # 等待畫面更新
        else:
            print("未偵測到每週獎勵畫面。")

    def distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def deduplicate(self, points, threshold=30):
        deduped = []
        for p in points:
            if all(self.distance(p, q) > threshold for q in deduped):
                deduped.append(p)
        return deduped

    def classify_shape(self, cnt):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area == 0 or perimeter == 0:
            return None
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter)

        if len(approx) == 3:
            return 'triangle'
        elif len(approx) == 4:
            pts = approx.reshape(4, 2)
            def angle(pt1, pt2, pt3):
                v1 = pt1 - pt2
                v2 = pt3 - pt2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                return np.arccos(cos_angle) * 180 / np.pi
            angles = [angle(pts[i], pts[(i+1)%4], pts[(i+2)%4]) for i in range(4)]
            if all(80 < a < 100 for a in angles):
                return 'square'
        elif 0.75 < circularity < 1.2 and len(approx) > 4:
            return 'circle'
        return None

    def find_shapes(self):
        min_area = 1500
        max_area = 5000

        screenshot, offset_x, offset_y = self.screenshot_game_area()
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circles, triangles, squares = [], [], []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            shape = self.classify_shape(cnt)
            if shape is None:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            x = int(M["m10"] / M["m00"]) + offset_x
            y = int(M["m01"] / M["m00"]) + offset_y
            center = (x, y)

            if shape == 'circle':
                circles.append(center)
            elif shape == 'triangle':
                triangles.append(center)
            elif shape == 'square':
                squares.append(center)

        circles = self.deduplicate(circles)
        triangles = self.deduplicate(triangles)
        squares = self.deduplicate(squares)

        return circles, triangles, squares

    def connect_station_auto(self):
        circles, triangles, squares = self.find_shapes()
        self.known_stations['circle'].update(circles)
        self.known_stations['triangle'].update(triangles)
        self.known_stations['square'].update(squares)

        all_stations = list(self.known_stations['circle'] | self.known_stations['triangle'] | self.known_stations['square'])
        all_routes = set(tuple(sorted(pair)) for pair in self.connected_pairs)

        candidates = []
        for station in all_stations:
            for connected_station in all_stations:
                if station != connected_station:
                    route = tuple(sorted([station, connected_station]))
                    if route not in all_routes:
                        candidates.append((station, connected_station))

        if candidates:
            station_A, station_B = max(candidates, key=lambda pair: self.distance(pair[0], pair[1]))
            self.connected_pairs.add(frozenset([station_A, station_B]))

            pyautogui.moveTo(*station_A, duration=0.3)
            pyautogui.mouseDown()
            time.sleep(0.3)
            pyautogui.moveTo(*station_B, duration=0.6)
            time.sleep(0.2)
            pyautogui.mouseUp()

            print(f"Connected: {station_A} → {station_B}")
        else:
            print("No valid connection found.")

    def reset(self):
        self.connected_pairs.clear()
        self.known_stations = {'circle': set(), 'triangle': set(), 'square': set()}

    def step(self, action):
        self.connect_station_auto()
        reward = 1
        done = False
        return None, reward, done
