import cv2

camera = cv2.VideoCapture(0)
print(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
print(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))


while True:
    ret, frame = camera.read()
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()