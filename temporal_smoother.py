from collections import deque, Counter

class TemporalSmoother:
    def __init__(self, history_len=15):
        self.history = {"Left": deque(maxlen=history_len),
                        "Right": deque(maxlen=history_len)}

    def smooth(self, hand_label, gesture):
        hist = self.history[hand_label]
        hist.append(gesture)
        most_common = Counter(hist).most_common(1)[0][0]
        return most_common
