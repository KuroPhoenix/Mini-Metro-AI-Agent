# Mini-Metro-AI-Agent

> What is our project's goal?

Reinforcement-learning agents that learn to play **[Mini Metro](https://dinopoloclub.com/minimetro/)** by perceiving the game purely from pixels (game frames of in-game screenshots) and planning underground lines like a human traffic engineer.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Dataset & Model Zoo](#dataset--model-zoo)
6. [Training Pipeline](#training-pipeline)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results](#results)
9. [Limitations & Future Work](#limitations--future-work)
10. [Contributing](#contributing)
11. [Team](#team)
12. [Reference](#reference)
13. [License](#license)

---

## Motivation

Urban transit design demands **constant re-planning under uncertainty in station occurances and passenger stream.**.  
Mini Metro compresses that challenge into an elegant puzzle: keep passengers moving while stations spawn randomly and resources are scarce.  
Our goal is to **teach an RL agent to survive as long as possible**—maximising passenger throughput before any station overcrowds. :contentReference[oaicite:1]{index=1}

---

## Key Features

| Module | What it does | Tech & Notes |
| ------ | ------------ | ----------- |
| **Pixel-to-world perception** | Detects station shapes, line colours, end-points and dotted segments directly from screenshots. | Roboflow 3.0 instance segmentation + RF-DETR (≈98 % precision on ~500 annotated frames). |
| **Action interface** | Converts discrete *PActions* (Action structure: [Verb-Arg-Arg2] e.g. DRAG REWARD to BLUE LINE”) into real mouse drags / key strokes. | `pyautogui`, Serpent.AI helper macros |
| **Reward Shaping** | Composite reward balances passenger flow, asset usage, line diversity, span, linearity and special-station coverage. | See `rl/reward.py` |
| **Learning Algorithm** | DAgger (Data Aggregation) with behaviour-cloning warm-start and iterative expert correction. | PyTorch 2.x |
| **Baseline agent** | Heuristic bot that recognises circle, triangle and square stations and lays naïve radial lines for quick benchmarking. | OpenCV contour analysis |


---

## Project Structure (Please refer to /New Final Project) 

```

Mini-Metro-AI-Agent/
├─ datasets/            # Roboflow exports + raw Serpent frames
├─ models/              # Trained weights (RF-DETR, line-segmenter, policies)
├─ rl/
│  ├─ dagger.py         # Main training loop
│  ├─ replay\_buffer.py
│  └─ reward.py
├─ serpent/             # Thin Serpent.AI wrapper to grab frames
├─ utils/               # Misc helpers (masking, OCR, logging)
├─ baseline/            # Baseline heuristic agent
├─ scripts/
│  ├─ run\_baseline.py
│  ├─ train\_detector.sh
│  └─ play\_with\_agent.py
└─ requirements.txt

````

*(Directories may change; check comments in the files for extra hints.)*

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/KuroPhoenix/Mini-Metro-AI-Agent.git
cd Mini-Metro-AI-Agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
````

### 2. Configure API keys

Roboflow inference requires a free API key. Export it once:

```bash
export ROBOFLOW_API_KEY="xxxxxxxxxxxxxxxxxxxx"
```
**CAUTION**
Our Roboflow model will expire on 6/5 2025. If you would like to train a CNN model yourself, we have provided Annotated Datasets that are compatitble with YOLOv12. 

### 3. Launch Mini Metro (Steam)

Set the game window to **1920 × 1080** for best detection accuracy.

### 4. Run the baseline bot

```bash
python scripts/run_baseline.py
```

### 5.Examine Detection Model Results

run `test.py` under New Final Project/ROI Capture/. Make sure to replace our Roboflow model with your custom-made model.


### 6. Train the agent with your expert plays

```bash
python -m RL.expert_demo --episodes 5 
```

**CAUTION** 
As of yet, our code cannot fully capture user's movements and map them to action space. Therefore it won't really catch anything.

---

## Dataset

| Asset                             | Location                | 
| --------------------------------- | ------------------------|
| **Station detector** (RF-DETR)    | `New Final Project/Annotation Dataset/MiniMetroStation`| 
| **Line-segmenter** (Roboflow 3.0) | `New Final Project/Annotation Dataset/Mini Metro Line Detect v2`  | 


*All assets are released for **research & educational use** only.*

---

## Training Pipeline

1. **Frame Capture** – Serpent.AI grabs raw frames at 30 Hz.
2. **Roboflow Detection** – Stations & lines fed through pre-trained detectors.
3. **State Extraction** – Build graph representation (nodes = stations, edges = line segments).
4. **Policy** – CNN-to-MLP predicts a *PAction* (`<line_id, op, station_id>`).
5. **DAgger Loop** – Interleave agent steps with on-the-fly expert correction, aggregating data.
6. **Optimisation** – Behaviour cloning loss + reward-weighted imitation.
7. **Deployment** – Trained policy drives `pyautogui` to execute actions in real time.

---

## Evaluation Metrics

| Metric                   | Why it matters                                             |
| ------------------------ | ---------------------------------------------------------- |
| **Passenger throughput** | Primary score – end-of-game passenger count.               |
| **Asset utilisation**    | Encourages using spare trains/lines strategically.         |
| **Line quality**         | Diversity, compact span, linearity, loop presence.         |
| **PR80 leaderboard**     | Aim: reach top-80 percentile on the in-game ranking list.  |


## Limitations & Future Work

* **Action-space pruning** – many invalid drag combinations still explored.
* **Temporal credit-assignment** – delayed station overcrowds hamper DAgger attempts.
* **Generalisation** – detectors tuned to Nanjing scene; other maps need finetuning.
* **Dataset size** – expert demonstrations currently small (< 1 hr).

See our presentation slides for an exhaustive roadmap.

---

## Contributing

Pull requests, bug reports and feature discussions are welcome!
Please open an issue first if you plan major changes.

---

## Team

| Name            | Student ID | Contribution (weight)                                                     |
| --------------- | ---------- | ------------------------------------------------------------------------- |
| **謝嘉宸 (Harry)** | 112550019  | 80 % – idea, perception, actions, RL pipeline, reward, env, repo & slides |
| **楊睿宸**         | 112550049  | 10 % – context/OCR classifiers, DQN baseline                              |
| **陳昱瑋**         | 112550041  | 10 % – traditional baseline, line detector                                |

*(Team 33, Introduction-to-AI Final Project, NYCU Spring 2025)*

---

## Reference

* Serpent.AI – [https://github.com/SerpentAI/SerpentAI](https://github.com/SerpentAI/SerpentAI)
* Roboflow – [https://roboflow.com](https://roboflow.com)
* OpenCV – [https://opencv.org](https://opencv.org)
* DAgger algorithm – [https://imitation.readthedocs.io/en/latest/algorithms/dagger.html](https://imitation.readthedocs.io/en/latest/algorithms/dagger.html)
* [Project slides](./docs/AI_Final_Project.pdf) for a full deep-dive.&#x20;

---

## License

This repository is released under the **MIT License**—see [`LICENSE`](LICENSE) for details.



