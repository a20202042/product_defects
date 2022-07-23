# ----------------------------------------透過pytesseract將圖片轉文字

from PIL import Image
import pytesseract
img_name = 'text_recognition_photo\\or_2.png' #Input圖片檔名
img = Image.open(img_name)
text = pytesseract.image_to_string(img, lang='eng') #辨識圖片中文字部分
print(text)

# 圖片辨識時，文字需擺正。目前測試可以辨識3D列印樣品文字，但實品雷雕文字因為太過模糊而無法辨識
# 三角形樣品目前測試僅有最小矩形裁切後的照片才能辨識，直接拍攝的原圖無法辨識出文字

# ----------------------------------------在圖片上依座標框出矩形

# import numpy as np
# import cv2 as cv
#
# image = cv.imread('text_recognition_photo\\test5.jpg')
#
# pt1, pt2 = (1000, 2000), (2000, 1000)
#
# text = 'test'
# fontFace = cv.FONT_HERSHEY_COMPLEX_SMALL
# fontScale = 1
# thickness = 1
# #繪製矩形框
# cv.rectangle(image, pt1, pt2, thickness = 2, color = (0, 255, 0))
# #算文本的寬高
# retval, baseLine = cv.getTextSize(text, fontFace = fontFace, fontScale = fontScale, thickness = thickness)
# #計算覆蓋文本矩形框座標
# topleft = (pt1[0], pt1[1] - retval[1])
# bottomright = (topleft[0] + retval[0], topleft[1] + retval[1])
# cv.rectangle(image, (topleft[0], topleft[1] - baseLine), bottomright, thickness = -1, color = (0, 255, 0))
# #繪製文本
# cv.putText(image, text, (pt1[0], pt1[1] - baseLine), fontScale = fontScale, fontFace = fontFace, thickness = thickness, color = (0, 0, 0))
# cv.imwrite('text_recognition_photo\\test5_2.jpg', image)

# ----------------------------------------將圖片中的文字以綠色矩形框選（目前只能辨識白底黑字）

# # 參考網址：https://blog.csdn.net/huobanjishijian/article/details/63685503
#
# import sys
#
# import cv2
# import numpy as np
#
# def preprocess(gray):
#     # 1. Sobel算子，x方向求梯度
#     sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
#     # 2. 二值化
#     ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
#
#     # 3. 膨脹和腐蝕操作的核函數
#     element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
#     element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
#
#     # 4. 膨脹一次，讓輪廓突出
#     dilation = cv2.dilate(binary, element2, iterations = 1)
#
#     # 5. 腐蝕一次，去掉細節，如表格線等。注意這裡去掉的是豎直的線
#     erosion = cv2.erode(dilation, element1, iterations = 1)
#
#     # 6. 再次膨脹，讓輪廓明顯一些
#     dilation2 = cv2.dilate(erosion, element2, iterations = 3)
#
#     # 7. 儲存中間圖片
#     cv2.imwrite("text_recognition_photo\\test\\binary.png", binary)
#     cv2.imwrite("text_recognition_photo\\test\\dilation.png", dilation)
#     cv2.imwrite("text_recognition_photo\\test\\erosion.png", erosion)
#     cv2.imwrite("text_recognition_photo\\test\\dilation2.png", dilation2)
#
#     return dilation2
#
# def findTextRegion(img):
#     region = []
#
#     # 1. 查找輪廓
#     contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 2. 篩選那些面積小的
#     for i in range(len(contours)):
#         cnt = contours[i]
#         # 計算該輪廓的面積
#         area = cv2.contourArea(cnt)
#
#         # 面積小的都篩選掉
#         if area < 1000:
#             continue
#
#         # 輪廓近似，作用很小
#         epsilon = 0.001 * cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, epsilon, True)
#
#         # 找到最小的矩形，該矩形可能有方向
#         rect = cv2.minAreaRect(cnt)
#         print("rect is:")
#         print(rect)
#
#         # box是四個點的座標
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)
#
#         # 計算高和寬
#         height = abs(box[0][1] - box[2][1])
#         width = abs(box[0][0] - [2][0])
#
#         # 篩選那些太細的矩形，留下扁的
#         if height > width * 1.2:
#             continue
#
#         region.append(box)
#
#     return region
#
# def detect(img):
#     # 1. 轉化成灰度圖
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # 2. 形態學變換的預處理，得到可以找到矩形的圖片
#     dilation = preprocess(gray)
#
#     # 3. 查找和篩選文字區域
#     region = findTextRegion(dilation)
#
#     # 4. 用綠線畫出這些找到的輪廓
#     for box in region:
#         cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
#
#     cv2.namedWindow("img", cv2.WINDOW_NORMAL)
#     cv2.imshow("img", img)
#
#     # 帶著輪廓的圖片
#     cv2.imwrite("text_recognition_photo\\test\\contours.png", img)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     imagePath = 'text_recognition_photo\\test10.png'
#     img = cv2.imread(imagePath)
#     detect(img)