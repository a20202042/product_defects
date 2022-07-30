import os, cv2
import numpy as np


# a = [(36, 136, 204, 23), (35, 111, 144, 22), (36, 89, 100, 20)]
# cnt = np.array(a)
# rect = cv2.minAreaRect(cnt)

def text_find(img):
    # img_name = 'match_data\\or_1.png'  # Input圖片檔名
    # img = cv2.imread(img_name)
    text = pytesseract.image_to_string(img, lang='eng')
    text = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", text)
    return text


def find_text_position():
    # 讀取圖片
    # imagePath = 'match_data\\or_2.png'
    imagePath = 'or_1_4.png'
    img = cv2.imread(imagePath)
    # 轉化成灰度圖
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 利用Sobel邊緣檢測生成二值圖
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 二值化
    ret, binary = cv2.threshold(sobel, 50, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # 膨脹、腐蝕
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 5))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 5))
    # 膨脹一次，讓輪廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)
    # 腐蝕一次，去掉細節
    erosion = cv2.erode(dilation, element1, iterations=1)
    # 再次膨脹，讓輪廓明顯一些
    dilation2 = cv2.dilate(erosion, element2, iterations=1)
    #  查詢輪廓和篩選文字區域
    region = []
    range_data = []
    contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        # 計算輪廓面積，並篩選掉面積小的
        area = cv2.contourArea(cnt)
        if (area < 600):
            continue
        # 找到最小的矩形
        rect = cv2.minAreaRect(cnt)
        # print("rect is: ")
        # print(rect)
        # box是四個點的座標
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 計算高和寬
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # 根據文字特徵，篩選那些太細的矩形，留下扁的
        if (height > width * 1.5):
            continue
        region.append(box)
    copy_img = img.copy()
    zero_shape = img.shape
    zero_img = np.zeros(shape=zero_shape, dtype='uint8')

    for box in region:
        # cv2.drawContours(copy_img, [box], 0, (0, 255, 0), 2)
        cv2.drawContours(zero_img, [box], 0, (0, 255, 0), 2)
        range_data.append(cv2.boundingRect(box))
        # print(box)
    for rect in range_data:
        range_ = 3
        crop_img = copy_img[rect[1] - range_:rect[1] + rect[3] + range_ * 2,
                   rect[0] - range_:rect[0] + rect[2] + range_ * 2]
        text = text_find(crop_img)
        print(text)
        cv2.imshow(str(text), crop_img)

    gray = cv2.cvtColor(zero_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), dtype=np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(zero_img, (x, y), (x + w, y + h), (255, 255, 0), 1)
    print(x, y, w, h)
    cv2.imshow('zero_img', zero_img)
    cv2.imshow('copy_img', copy_img)
    cv2.waitKey(0)


import pytesseract, re

find_text_position()
