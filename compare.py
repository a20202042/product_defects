from skimage.metrics import structural_similarity
import cv2
import numpy as np

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
    range_data = []
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
        range_data.append(cv2.boundingRect(box))
    for rect in range_data:
        print(rect)
    return range_data

# Load images
before = cv2.imread('match_data\\or_1.png')
before = cv2.resize(before, (250, 250), interpolation=cv2.INTER_AREA)


after = cv2.imread('or_1_2.png')
after = cv2.resize(after, (250, 250), interpolation=cv2.INTER_AREA)
range_data = find_text_position(after)


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
    if area > 350:
        x, y, w, h = cv2.boundingRect(c)
        for item in range_data:
            if x < item[0] < x+w and x < item[2] < x+w:
                print('123')
        cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
        cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)


cv2.imshow('before', before)
cv2.imshow('after', after)
cv2.imshow('filled after', filled_after)
cv2.waitKey()
