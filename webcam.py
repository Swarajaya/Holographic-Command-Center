import cv2

class Webcam:
    def __init__(self, width=640, height=480):
        self.cap = cv2.VideoCapture(0)
        self.width = width
        self.height = height

        if not self.cap.isOpened():
            raise Exception("Error: Could not open webcam")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Resize for performance
        frame = cv2.resize(frame, (self.width, self.height))

        # Flip vertically first if needed to correct upside-down webcams
        # frame = cv2.flip(frame, 0)

        # Flip horizontally to cancel default mirror effect
        frame = cv2.flip(frame, 1)  # this cancels the mirrored webcam effect

        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Test standalone webcam feed
if __name__ == "__main__":
    cam = Webcam()
    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        cv2.imshow("Webcam Feed", frame)

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
