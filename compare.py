from skimage.metrics import structural_similarity
import cv2, pytesseract, re
import numpy as np


def text_find(img):
    # img_name = 'match_data\\or_1.png'  # Input圖片檔名
    # img = cv2.imread(img_name)
    text = pytesseract.image_to_string(img, lang='eng')
    text = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", text)
    return text


def find_text_position(img):
    # 讀取圖片
    # imagePath = 'match_data\\or_3.png'
    # img = cv2.imread(imagePath)
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
        if (area < 500):
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
        cv2.drawContours(copy_img, [box], 0, (0, 255, 0), 2)
        cv2.drawContours(zero_img, [box], 0, (0, 255, 0), 2)
        range_data.append(cv2.boundingRect(box))
        # print(box)
    for rect in range_data:
        range_ = 3
        crop_img = copy_img[rect[1] - range_:rect[1] + rect[3] + range_ * 2,
                   rect[0] - range_:rect[0] + rect[2] + range_ * 2]
        text = text_find(crop_img)
        # print(text)
        # cv2.imshow(str(text), crop_img)

    gray = cv2.cvtColor(zero_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), dtype=np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(zero_img, (x, y), (x + w, y + h), (255, 255, 0), 1)
    print(x, y, w, h)
    return [x, y, w, h]


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
    range = 0
    x = min([left_point_x, right_point_x]) - range
    w = abs(right_point_x - left_point_x) + range * 2
    y = min([top_point_y, bottom_point_y]) - range
    h = abs(top_point_y - bottom_point_y) + range * 2
    crop_img = crop_img[y:y + h, x:x + w]
    return crop_img

def angle_calculate(box):
    import math
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
    import imutils
    rotated_img = imutils.rotate_bound(img, angle)
    # rotated_img
    return rotated_img

# Load images
before = cv2.imread('14526862-1.png')
before = cv2.resize(before, (240, 240), interpolation=cv2.INTER_AREA)

after = cv2.imread('14526862-1_compare.png')
after = cv2.resize(after, (240, 240), interpolation=cv2.INTER_AREA)

# crop_img, rotated_img = after.copy(), after.copy()
range_data = find_text_position(after)
#
# box_img, box_compact_img, box = box_compact(after)
#
# # # -----抓出最小方形角度並旋轉
# angle = angle_calculate(box)
# print(angle)
# rotated_img = rotate_img(rotated_img, angle)
# cv2.imshow('rotated_img', rotated_img)
# cv2.waitKey(0)
# #
# # # -----裁切出目標區域
# not_use, not_use_2, box = box_compact(rotated_img)
# crop_img = rotate_img(crop_img, angle)
# after = crop(np.min(box[:, 0]), np.max(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 1]), crop_img)
# after = cv2.resize(after, (240, 240), interpolation=cv2.INTER_AREA)

# Convert images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute SSIM between the two images
(score, diff) = structural_similarity(before_gray, after_gray, full=True)
print("Image Similarity: {:.4f}%".format(score * 100))

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1]
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] before we can use it with OpenCV
diff = (diff * 255).astype("uint8")
diff_box = cv2.merge([diff, diff, diff])

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 150:
        x, y, w, h = cv2.boundingRect(c)
        # print(x, y, w, h)
        # print(range_data)
        if range_data[0] < x < range_data[2] or range_data[0] < x + w < range_data[2]:
            pass
        # elif x < range_data[0] < x + w or x < range_data[2] < x + w:
        #     pass
        else:
            cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
            cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

cv2.imshow('before', before)
cv2.imshow('after', after)
cv2.imshow('filled after', filled_after)
cv2.waitKey()
