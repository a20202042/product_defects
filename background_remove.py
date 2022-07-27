import cv2
import numpy as np

img = cv2.imread('0726_test.png')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# adaptiveThresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 90)
# cv2.imshow('adaptiveThresh', adaptiveThresh)

low_green = np.array([0, 0, 90])
high_green = np.array([255, 255, 255])
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(imgHSV, low_green, high_green)
# mask = 255 - mask
res = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("img", img)
cv2.imshow("res", res)

cv2.waitKey(0)
