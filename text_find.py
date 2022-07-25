# -*- coding: utf-8 -*-

import cv2
import numpy as np

# 讀取圖片
imagePath = 'match_data\\or_3.png'
img = cv2.imread(imagePath)
# 轉化成灰度圖
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 利用Sobel邊緣檢測生成二值圖
sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
# 二值化
ret, binary = cv2.threshold(sobel, 50, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
# 膨脹、腐蝕
element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
# 膨脹一次，讓輪廓突出
dilation = cv2.dilate(binary, element2, iterations=1)
# 腐蝕一次，去掉細節
erosion = cv2.erode(dilation, element1, iterations=1)
# 再次膨脹，讓輪廓明顯一些
dilation2 = cv2.dilate(erosion, element2, iterations=2)
#  查詢輪廓和篩選文字區域
region = []
contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    cnt = contours[i]
    # 計算輪廓面積，並篩選掉面積小的
    area = cv2.contourArea(cnt)
    if (area < 1000):
        continue
    # 找到最小的矩形
    rect = cv2.minAreaRect(cnt)
    print("rect is: ")
    print(rect)
    # box是四個點的座標
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # 計算高和寬
    height = abs(box[0][1] - box[2][1])
    width = abs(box[0][0] - box[2][0])
    # 根據文字特徵，篩選那些太細的矩形，留下扁的
    if (height > width * 1.3):
        continue
    region.append(box)
copy_img = img.copy()
# 繪製輪廓
for box in region:
    cv2.drawContours(copy_img, [box], 0, (0, 255, 0), 2)
    c = cv2.boundingRect(box)
cv2.imshow('img', copy_img)


import cv2
from PIL import Image
import pytesseract
# # img_name = 'match_data\\or_1.png' #Input圖片檔名
# # # img = Image.open(img_name)
# # img = cv2.imread(img_name)
text = pytesseract.image_to_string(img, lang='eng')  # 辨識圖片中文字部分
print(text)
cv2.waitKey(0)
cv2.destroyAllWindows()
