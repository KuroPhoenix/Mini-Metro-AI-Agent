import cv2, json, glob
from plugins.SerpentMiniMetro_RL_AgentGameAgentPlugin.files.vision import detect_inventory

for f in glob.glob("screens/inv_*.png"):
    img = cv2.imread(f)
    print(f, detect_inventory(img))
