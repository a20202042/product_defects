# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('0.png')
# template = cv.imread('1.png')
# h, w, l = template.shape
#
# # 2 模板匹配
# # 2.1 模板匹配
# res = cv.matchTemplate(img, template, cv.TM_SQDIFF)
#
# # 2.2 返回图像中最匹配的位置，确定左上角的坐标，并将匹配位置绘制在图像上
# min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res) # 使用cv.minMaxLoc()搜索最匹配的位置。
# # 使用平方差时最小值为最佳匹配位置
# top_left = min_loc
# bottom_right = (top_left[0] + w, top_left[1] + h)
# cv.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
#
# # 3 图像显示
# plt.imshow(img[:, :, ::-1])
# plt.title('匹配结果'),
# plt.xticks([]),
# plt.yticks([])
# plt.show()

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('0.png',0)
img2 = img.copy()
template = cv.imread('test.jpg',0)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()