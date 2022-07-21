import cv2
import numpy as np
import math, imutils
import os

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 133.0)  # 亮度 130
cap.set(cv2.CAP_PROP_CONTRAST, 5.0)  # 对比度 32
cap.set(cv2.CAP_PROP_SATURATION, 83.0)  # 饱和度 64
cap.set(cv2.CAP_PROP_HUE, -1.0)  # 色调 0
cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)  # 曝光 -4
# print('亮度:', cap.get(cv2.CAP_PROP_BRIGHTNESS))
# print('对比度:', cap.get(cv2.CAP_PROP_CONTRAST))
# print('饱和度:', cap.get(cv2.CAP_PROP_SATURATION))
# print('色调:', cap.get(cv2.CAP_PROP_HUE))
# print('曝光度:', cap.get(cv2.CAP_PROP_EXPOSURE))

ret, cv_img = cap.read()
while True:
    # 擷取影像
    ret, img = cap.read()
    dir = 'match_data'
    files = os.listdir(dir)
    data = []
    for file in files:
        # print(dir + '\\' + file)
        template = cv2.imread(dir + '\\' + file)
        h, w, l = template.shape

        res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # print(min_val, max_val)
        max_val_ncc = '{:.3f}'.format(max_val)
        # print("correlation match score: " + max_val_ncc)

        if float(max_val_ncc) > 0.92:
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            # cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            # cv2.putText(img, 'text', top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            # cv2.imshow('target', img)
            data.append({'name': str(file), 'top_left': top_left, 'bottom_right': bottom_right})
            # cv2.imshow('target', img)
    # print(data)
    for item in data:
        # print(item['top_left'])
        cv2.rectangle(img, item['top_left'], item['bottom_right'], (0, 255, 0), 2)
        cv2.putText(img, item['name'], item['top_left'], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('target', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
