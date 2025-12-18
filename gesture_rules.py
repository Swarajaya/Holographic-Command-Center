import math

class GestureRules:
    def __init__(self):
        self.pinch_threshold = 40  # distance threshold for pinch
        self.fist_threshold = 100   # adjust according to webcam resolution

    def distance(self, p1, p2):
        return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

    def detect_gesture(self, landmarks, label):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        wrist = landmarks[0]
        tips = [landmarks[i] for i in [4,8,12,16,20]]

        # --- PINCH ---
        if self.distance(thumb_tip, index_tip) < self.pinch_threshold:
            return "PINCH"

        # --- OPEN_PALM ---
        # Check if fingers are far from wrist
        avg_dist = sum([self.distance(wrist, t) for t in tips]) / len(tips)
        if avg_dist > 150:
            return "OPEN_PALM"

        # --- FIST ---
        # Count how many fingers are near wrist
        fingers_folded = sum(1 for t in tips if self.distance(wrist, t) < self.fist_threshold)
        if fingers_folded >= 4:  # 4 or more fingers folded → FIST
            return "FIST"

        return "UNKNOWN"
