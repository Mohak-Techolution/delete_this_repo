import cv2

cap=cv2.VideoCapture(0)
while True:
    ret, frame=cap.read()
    if ret:
        cv2.imwrite(f"assets/stream_frame.jpg", frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1)==ord("x"):
            break

# observer design pattern>designguru.com
cap.release()
cv2.destroyAllWindows()