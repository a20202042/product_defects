import cv2
import numpy as np

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
import cv2

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
import cv2
import numpy as np

img = cv2.imread('1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), dtype=np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(cnt)
img_2 = img.copy()
cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
cv2.imshow('img', img)
# cv2.destroyAllWindows()
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img_2, [box], 0, (0, 0, 255), 2)
cv2.imshow('drawContours', img_2)
cv2.waitKey()
