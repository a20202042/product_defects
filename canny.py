import numpy as np
import cv2 as cv

lane = cv.imread('0530.jpg')
lane = cv.GaussianBlur(lane, (5, 5), 0)
lane = cv.Canny(lane, 200,200)
cv.imshow('lane', lane)
# cv.waitKey()

rho = 1 #距離解析度
theta = np.pi/180
threshold = 10
min_line_len = 20
max_line_grap = 20
lines = cv.HoughLinesP(lane, rho, theta, threshold, maxLineGap=max_line_grap)
line_img = np.zeros_like(lane)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv.line(line_img, (x1, y1), (x2, y2), 255, 1)
cv.imshow('lines_image', line_img)

o_img = cv.imread('0530.jpg')
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
gray_res = cv.morphologyEx(o_img, cv.MORPH_GRADIENT, kernel=kernel)
cv.imshow('gray_res', gray_res)

cv.waitKey()
# bgr = cv.imread('1.png')
# rgb = cv.cvtColor(cv.COLOR_BGR2RGB)
