 AI Holographic Command Center 🖐️✨

A **futuristic Iron-Man style AR control panel** fully controlled by hand gestures.  
This project combines **AI, Computer Vision, and AR UI** to create a real-time holographic interface.

---

## 🚀 Features

- **Hand Tracking:** Detects left and right hands with 21 landmarks (MediaPipe).  
- **Gesture Recognition:** Open palm, pinch, swipe, and more using AI & rule-based methods.  
- **State Machine:** Intelligent system behavior based on current gesture and mode.  
- **AR UI Panels:** Floating cards, circular HUDs, and real-time data overlays.  
- **Sci-Fi Effects:** Glow trails, depth-based scaling, parallax effects, and neon outlines.  
- **Interactions:**  
  - Pinch → Select  
  - Swipe → Switch panel/mode  
  - Palm movement → Scroll data  
  - Rotate/Zoom → Manipulate 3D objects

---

## 🛠️ Tech Stack

- **Python 3.10+**  
- **OpenCV** - Real-time video capture and UI overlay  
- **MediaPipe** - Hand landmark detection  
- **PyTorch** - Gesture classification model  
- **NumPy** - Matrix operations for gestures and panels  
- **Optional:** OpenGL for advanced 3D effects

---

## 🎯 How It Works

1. Capture webcam feed.  
2. Detect hands using MediaPipe.  
3. Extract landmarks and classify gestures.  
4. Feed gestures into a state machine.  
5. Render AR panels and HUDs with visual effects.  
6. Perform interaction based on gestures (select, scroll, zoom, rotate).

---

## 📁 Folder Structure

holo_command_center/
│
├── main.py
├── camera/
│ └── webcam.py
├── hand_tracking/
│ └── hand_detector.py
├── gestures/
│ ├── gesture_rules.py
│ ├── gesture_model.py
│ └── temporal_smoother.py
├── ui/
│ ├── panels.py
│ ├── huds.py
│ ├── glow.py
│ └── renderer.py
├── logic/
│ ├── state_machine.py
│ └── interaction.py
└── assets/

yaml
Copy code

---

## ⚡ How to Run

1. Clone the repo:
```bash
git clone https://github.com/yourusername/holo_command_center.git
cd holo_command_center
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the main program:

bash
Copy code
python main.py