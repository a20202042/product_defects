#小票水平分割
import cv2
import numpy as np

img = cv2.imread('text.jpg')
cv2.imshow("Orig Image", img)
# 輸出影象尺寸和通道資訊
sp = img.shape
print("影象資訊：", sp)
sz1 = sp[0]  # height(rows) of image
sz2 = sp[1]  # width(columns) of image
sz3 = sp[2]  # the pixels value is made up of three primary colors
print('width: %d \n height: %d \n number: %d' % (sz2, sz1, sz3))
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval, threshold_img = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("threshold_img", threshold_img)

# 水平投影分割影象
gray_value_x = []
for i in range(sz1):
    white_value = 0
    for j in range(sz2):
        if threshold_img[i, j] == 255:
            white_value += 1
    gray_value_x.append(white_value)
print("", gray_value_x)
# 建立影象顯示水平投影分割影象結果
hori_projection_img = np.zeros((sp[0], sp[1], 1), np.uint8)
for i in range(sz1):
    for j in range(gray_value_x[i]):
        hori_projection_img[i, j] = 255
cv2.imshow("hori_projection_img", hori_projection_img)
text_rect = []
# 根據水平投影分割識別行
inline_x = 0
start_x = 0
text_rect_x = []
for i in range(len(gray_value_x)):
    if inline_x == 0 and gray_value_x[i] > 10:
        inline_x = 1
        start_x = i
    elif inline_x == 1 and gray_value_x[i] < 10 and (i - start_x) > 5:
        inline_x = 0
        if i - start_x > 10:
            rect = [start_x - 1, i + 1]
            text_rect_x.append(rect)
print("分行區域，每行資料起始位置Y：", text_rect_x)
# 每行資料分段
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
dilate_img = cv2.dilate(threshold_img, kernel)
cv2.imshow("dilate_img", dilate_img)
cv2.waitKey(0)
for rect in text_rect_x:
    cropImg = dilate_img[rect[0]:rect[1],0:sp[1]]  # 裁剪影象y-start:y-end,x-start:x-end
    sp_y = cropImg.shape
    # 垂直投影分割影象
    gray_value_y = []
    for i in range(sp_y[1]):
        white_value = 0
        for j in range(sp_y[0]):
            if cropImg[j, i] == 255:
                white_value += 1
        gray_value_y.append(white_value)
    # 建立影象顯示水平投影分割影象結果
    veri_projection_img = np.zeros((sp_y[0], sp_y[1], 1), np.uint8)
    for i in range(sp_y[1]):
        for j in range(gray_value_y[i]):
            veri_projection_img[j, i] = 255
    cv2.imshow("veri_projection_img", veri_projection_img)
    # 根據垂直投影分割識別行
    inline_y = 0
    start_y = 0
    text_rect_y = []
    for i in range(len(gray_value_y)):
        if inline_y == 0 and gray_value_y[i] > 2:
            inline_y = 1
            start_y = i
        elif inline_y == 1 and gray_value_y[i] < 2 and (i - start_y) > 5:
            inline_y = 0
            if i - start_y > 10:
                rect_y = [start_y - 1, i + 1]
                text_rect_y.append(rect_y)
                text_rect.append([rect[0], rect[1], start_y - 1, i + 1])
                cropImg_rect = threshold_img[rect[0]:rect[1], start_y - 1:i + 1]  # 裁剪影象
                cv2.imshow("cropImg_rect", cropImg_rect)
                # cv2.imwrite("C:/Users/ThinkPad/Desktop/cropImg_rect.jpg",cropImg_rect)
                # break
        # break
# 在原圖上繪製截圖矩形區域
print("擷取矩形區域(y-start:y-end,x-start:x-end)：", text_rect)
rectangle_img = cv2.rectangle(img, (text_rect[0][2], text_rect[0][0]), (text_rect[0][3], text_rect[0][1]),
                              (255, 0, 0), thickness=1)
for rect_roi in text_rect:
    rectangle_img = cv2.rectangle(img, (rect_roi[2], rect_roi[0]), (rect_roi[3], rect_roi[1]), (255, 0, 0), thickness=1)
cv2.imshow("Rectangle Image", rectangle_img)

key = cv2.waitKey(0)
if key == 27:
    print(key)
    cv2.destroyAllWindows()