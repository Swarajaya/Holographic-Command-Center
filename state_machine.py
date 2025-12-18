class StateMachine:
    def __init__(self):
        self.state = "IDLE"  # initial state

    def update(self, gesture, hand_label=None):
        """
        Update the state based on current gesture
        Returns: new state and any action string
        """
        action = None

        if self.state == "IDLE":
            if gesture == "SWIPE":
                self.state = "PANEL_OPEN"
                action = "Open Panel"
            elif gesture == "PINCH":
                self.state = "ZOOM_MODE"
                action = "Enter Zoom Mode"

        elif self.state == "PANEL_OPEN":
            if gesture == "SWIPE":
                action = "Change Tab"
            elif gesture == "FIST":
                self.state = "IDLE"
                action = "Close Panel"

        elif self.state == "ZOOM_MODE":
            if gesture == "PINCH":
                self.state = "IDLE"
                action = "Exit Zoom Mode"

        # You can add more states like OBJECT_ROTATE, PANEL_SCROLL, etc.
        return self.state, action
