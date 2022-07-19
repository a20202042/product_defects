import cv2
import numpy as np
import math
import imutils


# bgr = cv2.imread('1.png')
# img = bgr.copy()
# gray_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray_img', gray_img)
# th, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
# contoure, he = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(bgr, contoure, -1, (0, 0, 255), 3)
# bounding_box = [cv2.boundingRect(cnt) for cnt in contoure]
#
# for bbox in bounding_box:
#     [x, y, w, h] = bbox
#     cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
# cv2.imshow('name', bgr)

# ------------

# import numpy as np
#
# img = cv2.imread("26_4.jpg")
#
# cv2.imshow("img1",img)
# # 轉換為灰度影象
# gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# rect= cv2.minAreaRect(contours[0])
# print(rect)
# points=cv2.boxPoints(rect)
# print(points)
# points=np.int0(points)
# print(points)
# cv2.drawContours(img,[points],0,(255,255,255),2)
#
# cv2.imshow("img2",img)
#
# cv2.waitKey()
# cv2.destroyAllWindows()
# ----------
def angle(main_down_center, main_top_center):
    if main_down_center[0] > main_top_center[0]:
        x = main_down_center[0] - main_top_center[0]
        y = main_down_center[1] - main_top_center[1]
    elif main_down_center[0] < main_top_center[0]:
        x = main_top_center[0] - main_down_center[0]
        y = main_top_center[1] - main_down_center[1]
    else:
        x = 0
        y = 0
    angle = -math.atan2(y, x) / math.pi * 180
    return angle


img = cv2.imread('1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), dtype=np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(cnt)
img_2, img_3 = img.copy(), img.copy()
cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
cv2.imshow('img', img)
# cv2.destroyAllWindows()
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img_2, [box], 0, (0, 0, 255), 2)

left_point_x = np.min(box[:, 0])
right_point_x = np.max(box[:, 0])
top_point_y = np.min(box[:, 1])
bottom_point_y = np.max(box[:, 1])
left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
print(left_point_y, right_point_y, top_point_x, bottom_point_x)
top_point = [top_point_x, top_point_y]
bottom_point = [bottom_point_x, bottom_point_y]
right_point = [right_point_x, right_point_y]
left_point = [left_point_x, left_point_y]
print(top_point, bottom_point, right_point, left_point)
point_data = [top_point, bottom_point, right_point, left_point]

# angle_ = angle(right_point, left_point)
angle_ = math.atan2(bottom_point[0] - right_point[0], bottom_point[1] - right_point[1]) / math.pi * 180
print(angle_)

rotated_img = imutils.rotate_bound(img_3, angle_)
cv2.imshow('drawContours', img_2)
cv2.imshow('rotated_img', rotated_img)
cv2.waitKey()
