# logic/interaction.py

class Interaction:
    def __init__(self):
        self.current_panel = 0
        self.zoom_level = 1.0
        self.rotation_angle = 0.0
        self.scroll_offset = 0.0

    def handle_gesture(self, gesture, hand_label, landmarks):
        """
        gesture: string, e.g., "PINCH", "SWIPE", "ROTATE", "PALM_UP", "TWO_HAND_SPREAD"
        hand_label: "Left" or "Right"
        landmarks: list of (x,y,z) for this hand
        """
        action = None

        if gesture == "PINCH":
            action = self.select_item(hand_label, landmarks)
        elif gesture == "SWIPE":
            action = self.switch_mode()
        elif gesture == "ROTATE":
            action = self.rotate_object(hand_label, landmarks)
        elif gesture == "PALM_UP" or gesture == "PALM_DOWN":
            action = self.scroll_panel(gesture, hand_label)
        elif gesture == "TWO_HAND_SPREAD":
            action = self.zoom_panel(hand_label, landmarks)

        return action

    # ----------------------------
    # Individual Actions
    # ----------------------------
    def select_item(self, hand_label, landmarks):
        # Example: return which panel/item was clicked
        return f"{hand_label} hand PINCH → select"

    def switch_mode(self):
        # Example: cycle through modes
        return "SWIPE → mode switch"

    def rotate_object(self, hand_label, landmarks):
        # Example: rotate based on wrist rotation
        self.rotation_angle += 5.0  # you can use landmarks to compute exact rotation
        return f"{hand_label} ROTATE → angle {self.rotation_angle:.1f}"

    def scroll_panel(self, gesture, hand_label):
        if gesture == "PALM_UP":
            self.scroll_offset += 10
        else:
            self.scroll_offset -= 10
        return f"{hand_label} scroll → offset {self.scroll_offset}"

    def zoom_panel(self, hand_label, landmarks):
        # Example: compute distance between both hands for zoom
        # landmarks is only one hand, you'll need both hand positions
        self.zoom_level += 0.1
        return f"Zoom level: {self.zoom_level:.1f}"
