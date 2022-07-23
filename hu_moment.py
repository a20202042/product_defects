import os

import cv2
from numpy import copysign, log10
import json


def hu_moment(im):
    # im = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
    _, im = cv2.threshold(im, 60, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(im)
    huMoments = cv2.HuMoments(moments)
    for i in range(0, 7):
        huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
        huMoments[i] = round(huMoments[i][0], 3)
    hu = []
    for item in huMoments:
        hu.append(float(item[0]))
    return hu


def rotate_img(img, angle):
    (h, w) = img.shape  # 讀取圖片大小
    center = (w // 2, h // 2)  # 找到圖片中心

    # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 第三個參數變化後的圖片大小
    rotate_img = cv2.warpAffine(img, M, (w, h))

    return rotate_img


all_data = {}
files = os.listdir('match_data')
for file in files:
    print(file)
    all_hu = []
    for i in range(0, 180):
        im = cv2.imread('match_data\\' + file, cv2.IMREAD_GRAYSCALE)
        im = rotate_img(im, i)
        hu = hu_moment(im)
        all_hu.append(hu)
        # print(hu)
    hu_1 = []
    hu_2 = []
    hu_3 = []
    for item in all_hu:
        hu_1.append(item[0])
        hu_2.append(item[1])
        hu_3.append(item[2])

    print('hu1:' + str(min(hu_1)) + ' ' + str(max(hu_1)))
    print('hu2:' + str(min(hu_2)) + ' ' + str(max(hu_2)))
    print('hu3:' + str(min(hu_3)) + ' ' + str(max(hu_3)))

    item_hu = {'hu1': [min(hu_1), max(hu_1)],
               'hu2': [min(hu_2), max(hu_2)],
               'hu3': [min(hu_3), max(hu_3)]}
    all_data.update({str(file):item_hu})

with open('data.json', 'w') as f:
    json.dump(all_data, f)

f = open('data.json')
data = json.load(f)
print(data)