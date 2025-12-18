import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, max_hands=2):
        self.max_hands = max_hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        hands_info = []

        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = handedness.classification[0].label
                lm_list = []
                for lm in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    lm_list.append((int(lm.x * w), int(lm.y * h)))

                # --- Sci-Fi Glow Skeleton ---
                for connection in self.mp_hands.HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = lm_list[start_idx]
                    end = lm_list[end_idx]
                    # Draw glowing line with layered transparency
                    for thickness, alpha in [(8,0.1),(5,0.3),(2,1.0)]:
                        overlay = frame.copy()
                        cv2.line(overlay, start, end, (0,255,0), thickness)
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Draw glowing circles at landmarks
                for point in lm_list:
                    for radius, alpha in [(6,0.1),(4,0.3),(2,1.0)]:
                        overlay = frame.copy()
                        cv2.circle(overlay, point, radius, (0,255,0), -1)
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                hands_info.append({"label": label, "landmarks": lm_list})

        return frame, hands_info
