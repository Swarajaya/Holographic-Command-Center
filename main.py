import cv2
import numpy as np
from hand_tracking.hand_detector import HandDetector
from collections import deque, Counter

# ----------------------------
# Initialize Hand Detector
# ----------------------------
detector = HandDetector(max_hands=2)

# ----------------------------
# Webcam
# ----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# ----------------------------
# Hand trails
# ----------------------------
hand_trails = {"Left": [], "Right": []}
max_trail_len = 5
connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
               (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
               (0,17),(17,18),(18,19),(19,20)]

# ----------------------------
# Gesture temporal smoothing
# ----------------------------
gesture_history = {"Left": deque(maxlen=10), "Right": deque(maxlen=10)}
def smooth_gesture(label, pred):
    gesture_history[label].append(pred)
    most_common = Counter(gesture_history[label]).most_common(1)
    return most_common[0][0] if most_common else pred

# ----------------------------
# State Machine
# ----------------------------
state = "IDLE"
def update_state(current_state, gesture):
    if current_state == "IDLE":
        if gesture == "Swipe":
            return "PANEL_OPEN"
    elif current_state == "PANEL_OPEN":
        if gesture == "Swipe":
            return "SCROLL"
        if gesture == "Pinch":
            return "OBJECT_ROTATE"
    elif current_state == "SCROLL":
        if gesture == "Fist":
            return "IDLE"
    return current_state

# ----------------------------
# UI Functions
# ----------------------------
def draw_floating_panel(frame, x, y, w=120, h=60, color=(50,200,255), alpha=0.3):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x,y), (x+w, y+h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    return frame

def draw_circular_hud(frame, x, y, radius=30, color=(50,200,255)):
    cv2.circle(frame, (x, y), radius, color, 2)
    cv2.circle(frame, (x, y), radius-10, color, 1)
    return frame

def draw_progress_ring(frame, x, y, radius=40, progress=0.5, color=(50,200,255), thickness=3):
    cv2.ellipse(frame, (x, y), (radius, radius), 0, -90, int(progress*360-90), color, thickness)
    return frame

def add_glow(frame, overlay, alpha=0.15, blur_size=(3,3)):
    kx = blur_size[0] if blur_size[0] % 2 == 1 else blur_size[0]+1
    ky = blur_size[1] if blur_size[1] % 2 == 1 else blur_size[1]+1
    glow = cv2.GaussianBlur(overlay, (kx, ky), 0)
    cv2.addWeighted(glow, alpha, frame, 1-alpha, 0, frame)
    return frame

# ----------------------------
# Rule-Based Gesture Detection
# ----------------------------
def detect_gesture(landmarks):
    """
    landmarks: list of (x, y)
    Returns: 'Pinch', 'Open Palm', 'Fist', 'Swipe'
    """
    if not landmarks or len(landmarks) < 21:
        return "Unknown"

    def distance(p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]

    # Pinch: thumb & index tip close
    if distance(thumb_tip, index_tip) < 40:
        return "Pinch"

    # Open Palm: all fingers far from wrist
    finger_tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    if all(distance(wrist, tip) > 80 for tip in finger_tips):
        return "Open Palm"

    # Fist: all fingers close to wrist
    if all(distance(wrist, tip) < 60 for tip in finger_tips):
        return "Fist"

    # Swipe: if palm moves fast horizontally
    # We'll skip real velocity calc for simplicity, can be added later
    return "Unknown"

# ----------------------------
# Main Loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    overlay = np.zeros_like(frame)

    frame, hands = detector.find_hands(frame)

    for hand in hands:
        label = hand.get("label", "Left")
        landmarks = hand.get("landmarks", [])

        if not landmarks:
            continue

        # Landmarks in pixels
        landmarks_px = [(int(l[0]), int(l[1])) for l in landmarks]

        # ----------------------------
        # Hand trails
        # ----------------------------
        x0, y0 = landmarks_px[0]
        hand_trails[label].append((x0, y0))
        if len(hand_trails[label]) > max_trail_len:
            hand_trails[label].pop(0)
        for i in range(1, len(hand_trails[label])):
            alpha_trail = i / len(hand_trails[label])
            color = (0, int(255*alpha_trail), 255)
            cv2.line(overlay, hand_trails[label][i-1], hand_trails[label][i], color, 2)

        # ----------------------------
        # Skeleton
        # ----------------------------
        for lm in landmarks_px:
            cv2.circle(overlay, lm, 5, (0,255,255), -1)
        for a,b in connections:
            cv2.line(overlay, landmarks_px[a], landmarks_px[b], (0,255,128), 2)

        # ----------------------------
        # Panels & HUD
        # ----------------------------
        x_offset = -50 if label=="Left" else 50
        y_offset = -50
        x_panel, y_panel = x0 + x_offset, y0 + y_offset

        frame = draw_floating_panel(frame, x_panel, y_panel)
        frame = draw_circular_hud(frame, x0 + 60, y0 - 50)
        frame = draw_progress_ring(frame, x0, y0 - 100, progress=0.6)

        # ----------------------------
        # Detect gestures (rule-based)
        # ----------------------------
        gesture = detect_gesture(landmarks_px)
        gesture = smooth_gesture(label, gesture)

        # Update state
        state = update_state(state, gesture)

        # ----------------------------
        # Display info
        # ----------------------------
        text_x = max(5, min(frame.shape[1]-150, x_panel))
        text_y = max(20, y_panel-10)
        cv2.putText(frame, f"{label} Hand", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Gesture: {gesture}", (text_x, text_y+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"State: {state}", (text_x, text_y+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,128,255), 2)

    # ----------------------------
    # Glow overlay
    # ----------------------------
    frame = add_glow(frame, overlay, alpha=0.15, blur_size=(3,3))

    cv2.imshow("Holographic Command Center", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
