## In Mini_Metro_Env:

```python
    # def find_shapes(self):
    #     # This function returns annotated image stations and box coords for each station
    #     annot_img, boxes = annotate_stations(self.screenshot)
    #     return annot_img, boxes
    #
    # def find_lines(self):
    #     annot_img, polygons = annotate_lines(self.screenshot, conf=0.5, overlap=0.4)
    #     return annot_img, polygons
    #
    # def get_context(self):
    #     ctx, reward = classify_context(self.screenshot)
    #     return ctx, reward
    # def distance(self, p1, p2):
    #     return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    # def reset(self):
    #     self.connected_pairs.clear()
    #     self.known_stations = {'circle': set(), 'triangle': set(), 'square': set()}
    #
    # def step(self, action):
    #     self.connect_station_auto()
    #     reward = 1
    #     done = False
    #     return None, reward, done
    #
    #
    # def connect_station_auto(self):
    #     circles, triangles, squares = self.find_shapes()
    #
    #     print(f"Detected circles: {len(circles)} at {circles}")
    #     print(f"Detected triangles: {len(triangles)} at {triangles}")
    #     print(f"Detected squares: {len(squares)} at {squares}")
    #
    #     new_circles = [c for c in circles if c not in self.known_stations['circle']]
    #     new_triangles = [t for t in triangles if t not in self.known_stations['triangle']]
    #     new_squares = [s for s in squares if s not in self.known_stations['square']]
    #
    #     print(f"New circles: {len(new_circles)} at {new_circles}")
    #     print(f"New triangles: {len(new_triangles)} at {new_triangles}")
    #     print(f"New squares: {len(new_squares)} at {new_squares}")
    #
    #     self.known_stations['circle'].update(circles)
    #     self.known_stations['triangle'].update(triangles)
    #     self.known_stations['square'].update(squares)
    #
    #     candidates = []
    #
    #     # 新站優先
    #     for c in new_circles:
    #         for t in triangles:
    #             candidates.append((c, t))
    #         for s in squares:
    #             candidates.append((c, s))
    #
    #     for t in new_triangles:
    #         for c in circles:
    #             candidates.append((t, c))
    #         for s in squares:
    #             candidates.append((t, s))
    #
    #     for s in new_squares:
    #         for c in circles:
    #             candidates.append((s, c))
    #         for t in triangles:
    #             candidates.append((s, t))
    #
    #     # 若沒新站候選，加入所有包含圓形的組合
    #     if not candidates:
    #         for c in circles:
    #             for t in triangles:
    #                 candidates.append((c, t))
    #             for s in squares:
    #                 candidates.append((c, s))
    #
    #         for t in triangles:
    #             for s in squares:
    #                 candidates.append((t, s))
    #
    #     # 過濾已連線組合
    #     candidates = [pair for pair in candidates if frozenset(pair) not in self.connected_pairs]
    #
    #     print(f"Candidates after filtering duplicates: {len(candidates)}")
    #     for pair in candidates:
    #         dist = self.distance(pair[0], pair[1])
    #         contains_circle = any(station in self.known_stations['circle'] for station in pair)
    #         print(f"Pair {pair} distance: {dist:.1f}, contains_circle: {contains_circle}")
    #
    #     if not candidates:
    #         print("No line co")
    #         return
    #
    #     # 優先挑包含圓形組合中距離最大的
    #     circle_pairs = [pair for pair in candidates if any(station in self.known_stations['circle'] for station in pair)]
    #
    #     if circle_pairs:
    #         station_A, station_B = max(circle_pairs, key=lambda pair: self.distance(pair[0], pair[1]))
    #         print(f"Selected circle pair: {station_A} → {station_B}")
    #     else:
    #         station_A, station_B = max(candidates, key=lambda pair: self.distance(pair[0], pair[1]))
    #         print(f"Selected non-circle pair: {station_A} → {station_B}")
    #
    #     self.connected_pairs.add(frozenset([station_A, station_B]))
    #
    #     pyautogui.moveTo(*station_A, duration=0.3)
    #     pyautogui.mouseDown()
    #     time.sleep(0.3)
    #     pyautogui.moveTo(*station_B, duration=0.6)
    #     time.sleep(0.2)
    #     pyautogui.mouseUp()
    #
    #     print(f"Connected: {station_A} → {station_B}")
    #
    #

```