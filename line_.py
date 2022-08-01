import cv2
import numpy as np
import math, imutils, json, os
from math import copysign, log10
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
from PIL import Image
import pytesseract, re

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 133.0)  # 亮度 130
cap.set(cv2.CAP_PROP_CONTRAST, 5.0)  # 对比度 32
cap.set(cv2.CAP_PROP_SATURATION, 83.0)  # 饱和度 64
cap.set(cv2.CAP_PROP_HUE, -1.0)  # 色调 0
cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)  # 曝光 -4

ret, cv_img = cap.read()
# line_pix_data = {}
# for i_1 in range(255, 500):
#     for i_2 in range(255, 256):
#         (b, g, r) = cv_img[i_1, i_2]
#         # cv_img[i_1, i_2] = (255, 255, 255)
#         line_pix_data.update({str(i_1) + ',' + str(i_2): (0, 0, 0)})
check = False
one_check = True

number = 2
line_pix_data = {}
for i_1 in range(255, 500):
    for i_2 in range(255, 256):
        # cv_img[i_1, i_2] = (255, 255, 255)
        line_pix_data.update({str(i_1) + ',' + str(i_2): (0, 0, 0)})

def box_compact(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    img_2, img_3 = img.copy(), img.copy()
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
    # cv2.imshow('img', img)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img_2, [box], 0, (0, 0, 255), 2)
    return img, img_2, box

def crop(left_point_x, right_point_x, top_point_y, bottom_point_y, crop_img):
    range = 3
    x = min([left_point_x, right_point_x]) - range
    w = abs(right_point_x - left_point_x) + range * 2
    y = min([top_point_y, bottom_point_y]) - range
    h = abs(top_point_y - bottom_point_y) + range * 2
    crop_img = crop_img[y:y + h, x:x + w]
    return crop_img


while (True):

    # 擷取影像
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    frame_2 = frame.copy()
    # (b, g, r) = frame[0, 0]
    # frame[0, 0] = (0, 0, 0)
    for i_1 in range(255, 500):
        for i_2 in range(255, 256):
            frame_2[i_1, i_2] = (255, 255, 255)
            # line_pix_data.update({str(i_1) + ',' + str(i_2): (0, 0, 0)})
    # cv2.line(frame_2, (255, 256), (255, 500), (255, 255, 0), 3)


    low_green = np.array([0, 0, 90])
    high_green = np.array([255, 255, 255])
    imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, low_green, high_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # cv2.imshow('cam', frame)
    for item in line_pix_data.keys():
        position_1, position_2 = item.split(',')
        (b, g, r) = res[int(position_1), int(position_2)]
        if (b, g, r) != (0, 0, 0):
            cv2.putText(frame_2, 'insert', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            check = True
            break
        elif (b, g, r) == (0, 0, 0):
            one_check = True
            check = False
            break

    if check is True and one_check is True:
        one_check = False
        #box標記
        status = cv2.imwrite('background_remove.png', res)
        box_img, box_compact_img, box = box_compact(res)
        crop_img = crop(np.min(box[:, 0]), np.max(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 1]), res)
        print(crop_img.shape)
        cv2.imshow('crop_img', crop_img)

    cv2.imshow('cam', frame_2)
    cv2.imshow('res', res)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
