import numpy as np
import random
import cv2
import pyautogui
import time
import math
from collections import defaultdict

class MiniMetroRLAgent:
    def __init__(self, actions):
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.actions = actions
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.2

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(len(self.actions)))
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - predict)

class MiniMetroEnv:
    def __init__(self):
        self.known_stations = {'circle': set(), 'triangle': set(), 'square': set()}
         # 記錄已連線過的車站對
        self.connected_pairs = set()  
       

    def screenshot_game_area(self):
        x, y, width, height = 90, 85, 1280, 800
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        return screenshot, x, y

    def distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def deduplicate(self, points, threshold=30):
        deduped = []
        for p in points:
            if all(self.distance(p, q) > threshold for q in deduped):
                deduped.append(p)
        return deduped

    def find_shapes(self):
        min_area = 2000
        max_area = 5000

        screenshot, offset_x, offset_y = self.screenshot_game_area()
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        # 找三角形和正方形
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        triangles, squares = [], []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            x = int(M["m10"] / M["m00"]) + offset_x
            y = int(M["m01"] / M["m00"]) + offset_y
            center = (x, y)

            if len(approx) == 3:
                triangles.append(center)
            elif len(approx) == 4:
                squares.append(center)

        # 用 HoughCircles 找圓形
        circles = []
        circles_cv = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=15,
            maxRadius=40
        )
        if circles_cv is not None:
            circles_cv = np.round(circles_cv[0, :]).astype("int")
            for (x, y, r) in circles_cv:
                # 確認圓形半徑及中心在遊戲區域範圍
                if min_area < math.pi * r * r < max_area:
                    circles.append((x + offset_x, y + offset_y))

        # 去除重複點
        circles = self.deduplicate(circles)
        triangles = self.deduplicate(triangles)
        squares = self.deduplicate(squares)

        return circles, triangles, squares


    def connect_station_auto(self):
        circles, triangles, squares = self.find_shapes()

        print(f"Detected circles: {len(circles)} at {circles}")
        print(f"Detected triangles: {len(triangles)} at {triangles}")
        print(f"Detected squares: {len(squares)} at {squares}")

        new_circles = [c for c in circles if c not in self.known_stations['circle']]
        new_triangles = [t for t in triangles if t not in self.known_stations['triangle']]
        new_squares = [s for s in squares if s not in self.known_stations['square']]

        print(f"New circles: {len(new_circles)} at {new_circles}")
        print(f"New triangles: {len(new_triangles)} at {new_triangles}")
        print(f"New squares: {len(new_squares)} at {new_squares}")

        self.known_stations['circle'].update(circles)
        self.known_stations['triangle'].update(triangles)
        self.known_stations['square'].update(squares)

        candidates = []

        # 新站優先
        for c in new_circles:
            for t in triangles:
                candidates.append((c, t))
            for s in squares:
                candidates.append((c, s))

        for t in new_triangles:
            for c in circles:
                candidates.append((t, c))
            for s in squares:
                candidates.append((t, s))

        for s in new_squares:
            for c in circles:
                candidates.append((s, c))
            for t in triangles:
                candidates.append((s, t))

        # 若沒新站候選，加入所有包含圓形的組合
        if not candidates:
            for c in circles:
                for t in triangles:
                    candidates.append((c, t))
                for s in squares:
                    candidates.append((c, s))

            for t in triangles:
                for s in squares:
                    candidates.append((t, s))

        # 過濾已連線組合
        candidates = [pair for pair in candidates if frozenset(pair) not in self.connected_pairs]

        print(f"Candidates after filtering duplicates: {len(candidates)}")
        for pair in candidates:
            dist = self.distance(pair[0], pair[1])
            contains_circle = any(station in self.known_stations['circle'] for station in pair)
            print(f"Pair {pair} distance: {dist:.1f}, contains_circle: {contains_circle}")

        if not candidates:
            print("No line co")
            return

        # 優先挑包含圓形組合中距離最大的
        circle_pairs = [pair for pair in candidates if any(station in self.known_stations['circle'] for station in pair)]

        if circle_pairs:
            station_A, station_B = max(circle_pairs, key=lambda pair: self.distance(pair[0], pair[1]))
            print(f"Selected circle pair: {station_A} → {station_B}")
        else:
            station_A, station_B = max(candidates, key=lambda pair: self.distance(pair[0], pair[1]))
            print(f"Selected non-circle pair: {station_A} → {station_B}")

        self.connected_pairs.add(frozenset([station_A, station_B]))

        pyautogui.moveTo(*station_A, duration=0.3)
        pyautogui.mouseDown()
        time.sleep(0.3)
        pyautogui.moveTo(*station_B, duration=0.6)
        time.sleep(0.2)
        pyautogui.mouseUp()

        print(f"Connected: {station_A} → {station_B}")


    def reset(self):
        self.connected_pairs.clear()
        self.known_stations = {'circle': set(), 'triangle': set(), 'square': set()}

    def step(self, action):
        self.connect_station_auto()
        reward = 1
        done = False
        return None, reward, done


