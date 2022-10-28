import cv2

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture()
cap.open(0)
# The device number might be 0 or 1 depending on the device and the webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)

while (True):
    ret, frame = cap.read()
    print("FPS: " + str(cap.get(cv2.CAP_PROP_FPS)))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 133.0)  # 亮度 130
# cap.set(cv2.CAP_PROP_CONTRAST, 5.0)  # 对比度 32
# cap.set(cv2.CAP_PROP_SATURATION, 83.0)  # 饱和度 64
# cap.set(cv2.CAP_PROP_HUE, -1.0)  # 色调 0
# cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)  # 曝光 -4
# # print('亮度:', cap.get(cv2.CAP_PROP_BRIGHTNESS))
# # print('对比度:', cap.get(cv2.CAP_PROP_CONTRAST))
# # print('饱和度:', cap.get(cv2.CAP_PROP_SATURATION))
# # print('色调:', cap.get(cv2.CAP_PROP_HUE))
# # print('曝光度:', cap.get(cv2.CAP_PROP_EXPOSURE))
#
# ret, cv_img = cap.read()
