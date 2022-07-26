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
# print('亮度:', cap.get(cv2.CAP_PROP_BRIGHTNESS))
# print('对比度:', cap.get(cv2.CAP_PROP_CONTRAST))
# print('饱和度:', cap.get(cv2.CAP_PROP_SATURATION))
# print('色调:', cap.get(cv2.CAP_PROP_HUE))
# print('曝光度:', cap.get(cv2.CAP_PROP_EXPOSURE))

ret, cv_img = cap.read()


def contour_calculation(size, frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    gray_res = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel=kernel)
    return gray_res


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
    return img, img_2, box, rect


def angle_calculate(box):
    # print(point_1, point_2)
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])
    left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
    right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
    top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
    bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
    # print(left_point_y, right_point_y, top_point_x, bottom_point_x)
    top_point = [top_point_x, top_point_y]
    bottom_point = [bottom_point_x, bottom_point_y]
    right_point = [right_point_x, right_point_y]
    left_point = [left_point_x, left_point_y]
    # print(top_point, bottom_point, right_point, left_point)
    point_data = [top_point, bottom_point, right_point, left_point]
    # angle_ = angle(right_point, left_point)
    angle_ = math.atan2(bottom_point[0] - right_point[0], bottom_point[1] - right_point[1]) / math.pi * 180
    return angle_


def rotate_img(img, angle):
    rotated_img = imutils.rotate_bound(img, angle)
    # rotated_img
    return rotated_img


def crop(left_point_x, right_point_x, top_point_y, bottom_point_y, crop_img):
    range = 0
    x = min([left_point_x, right_point_x]) - range
    w = abs(right_point_x - left_point_x) + range * 2
    y = min([top_point_y, bottom_point_y]) - range
    h = abs(top_point_y - bottom_point_y) + range * 2
    crop_img = crop_img[y:y + h, x:x + w]
    return crop_img


def subimage(image, center, theta, width, height):
    '''
    Rotates OpenCV image around center with angle theta (in deg)
    then crops the image according to width and height.
    '''

    # Uncomment for theta in radians
    # theta *= 180/np.pi

    shape = (image.shape[1], image.shape[0])  # cv2.warpAffine expects shape in (length, height)

    matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
    image = cv2.warpAffine(src=image, M=matrix, dsize=shape)

    x = int(center[0] - width / 2)
    y = int(center[1] - height / 2)

    image = image[y:y + height, x:x + width]

    return image


import cv2
import numpy as np


def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
               pts[1][0]:pts[2][0]]

    return img_crop

def getSubImage(rect, src):
    # Get center, size, and angle from rect
    center, size, theta = rect
    # Convert to int
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D( center, theta, 1)
    # Perform rotation on src image
    dst = cv2.warpAffine(src, M, src.shape[:2])
    out = cv2.getRectSubPix(dst, size, center)
    return out

while (True):
    ret, frame = cap.read()
    crop_img = frame.copy()
    cv2.imshow('cam', frame)

    gray_res = contour_calculation(2, frame)
    # cv2.imshow('gray_res', gray_res)

    box_img, box_compact_img, box, rect = box_compact(frame)
    cv2.imshow('box_compact_img', box_compact_img)
    # crop_img = crop(np.min(box[:, 0]), np.max(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 1]), crop_img)
    # cv2.imshow('crop_img', crop_img)
    # # # # -----抓出最小方形角度並旋轉
    # angle = angle_calculate(box)
    # # if angle >= 90:
    # #     angle = angle - 90
    # rotated_img = rotate_img(crop_img, angle)
    # cv2.imshow('rotated_img', rotated_img)
    # # #
    # # # # -----裁切出目標區域
    # test = crop_img.copy()
    # not_use, not_use_2, box = box_compact(rotated_img)
    # crop_img = rotate_img(crop_img, angle)
    # crop_img = crop(np.min(box[:, 0]), np.max(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 1]), crop_img)
    # cv2.imshow('rotated_img', rotated_img)
    # cv2.imshow('crop_img', crop_img)
    out = getSubImage(rect, crop_img)
    cv2.imshow('out', out)
    # Save image
    # cv2.imwrite('out.jpg', out)
    if cv2.waitKey(1) == ord('q'):
        break
