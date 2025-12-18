
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW on Windows
if not cap.isOpened():
    print("Cannot open webcam. Make sure no other app is using it!")
    exit()
else:
    print("Webcam opened successfully.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue
    cv2.imshow("Webcam Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
