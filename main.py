import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280 / 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720 / 2)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 133.0)  # 亮度 130
cap.set(cv2.CAP_PROP_CONTRAST, 5.0)  # 对比度 32
cap.set(cv2.CAP_PROP_SATURATION, 83.0)  # 饱和度 64
cap.set(cv2.CAP_PROP_HUE, -1.0)  # 色调 0
cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)  # 曝光 -4
print('亮度:', cap.get(cv2.CAP_PROP_BRIGHTNESS))
print('对比度:', cap.get(cv2.CAP_PROP_CONTRAST))
print('饱和度:', cap.get(cv2.CAP_PROP_SATURATION))
print('色调:', cap.get(cv2.CAP_PROP_HUE))
print('曝光度:', cap.get(cv2.CAP_PROP_EXPOSURE))

ret, cv_img = cap.read()
while (True):
    # 擷取影像
    ret, frame = cap.read()
    cv2.imshow('cam', frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gray_res = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel=kernel)
    cv2.imshow('gray_res', gray_res)

    bgr = gray_res
    gray_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    th, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
    contoure, he = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_box = [cv2.boundingRect(cnt) for cnt in contoure]

    for bbox in bounding_box:
        [x, y, w, h] = bbox
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('name', bgr)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
