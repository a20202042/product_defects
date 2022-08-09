import os

import cv2
from numpy import copysign, log10
import json
import numpy as np


def hu_moment(im):
    # im = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
    _, im = cv2.threshold(im, 60, 255, cv2.THRESH_BINARY)
    cv2.imshow('im', im)
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


# def circle_detect(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉灰階
#
#     gray = cv2.medianBlur(gray, 11)  # 使用中值平滑對影圖像進行模糊處理
#
#     rows = gray.shape[0]
#
#     circles = cv2.HoughCircles(gray,
#                                cv2.HOUGH_GRADIENT,
#                                minDist=40,
#                                # 圓心距離
#                                dp=1.2,
#                                # 檢測圓心的累加器精度和圖像精度比的倒數(1=相同分辨綠，2=累加器是輸入圖案一半大的寬高)
#                                param1=150,
#                                # canny檢測的高闊值，低闊值為一半
#                                param2=50,
#                                # 圓心的累加器闊值，越小檢測更多的圓，越大越精確
#                                minRadius=1,
#                                # 最小半徑
#                                maxRadius=40)
#     # 最大半徑
#     circles_data = []
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         # 四捨五入(np.around)後，轉換成16位無符號整數(np.uint16)。
#         number = 1
#         for i in circles[0, :]:
#             center = (i[0], i[1])
#             # circle center
#             cv2.circle(img, center, 1, (0, 100, 100), 5)
#             # (原圖、圓心、半徑、顏色、粗細)
#             # circle outline
#             radius = i[2]
#             cv2.circle(img, center, radius, (255, 255, 255), 5)
#             circles_data.append({'number': number, 'center': center, 'r': i})  # 每個圈的數據
#             number = number + 1  # 計數器加1
#     # print(circles_data)
#     for item in circles_data:
#         text = '%s' % item['number']
#         cv2.putText(img, text, item['center'], cv2.FONT_HERSHEY_SIMPLEX,
#                     3, (0, 0, 255), 5, cv2.LINE_AA)
#         # 繪製文字(圖片影像/繪製的文字/左上角坐標/字體/字體大小/顏色/字體粗細/字體線條種類)
#     # cv.imshow("detected circles", src)
#     #
#     # cv.waitKey(0)
#     return img, circles_data



all_data = {}
files = os.listdir('match_data')
for file in files:
    # img = cv2.imread('match_data\\' + file)
    # img, circles_data = circle_detect(img)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
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
    all_data.update({str(file): item_hu})

with open('data.json', 'w') as f:
    json.dump(all_data, f)

f = open('data.json')
data = json.load(f)
print(data)
