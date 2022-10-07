import sys
import cv2 as cv
import numpy as np


def main(img):
    # default_file = 'or_1_2.png'
    # src = cv.imread(img) #讀取圖片
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 轉灰階

    gray = cv.medianBlur(gray, 9)  # 使用中值平滑對影圖像進行模糊處理

    rows = gray.shape[0]

    circles = cv.HoughCircles(gray,
                              cv.HOUGH_GRADIENT,
                              minDist=40,
                              # 圓心距離
                              dp=1.2,
                              # 檢測圓心的累加器精度和圖像精度比的倒數(1=相同分辨綠，2=累加器是輸入圖案一半大的寬高)
                              param1=150,
                              # canny檢測的高闊值，低闊值為一半
                              param2=35,
                              # 圓心的累加器闊值，越小檢測更多的圓，越大越精確
                              minRadius=1,
                              # 最小半徑
                              maxRadius=40)
    # 最大半徑
    circles_data = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # 四捨五入(np.around)後，轉換成16位無符號整數(np.uint16)。
        number = 1
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(img, center, 1, (0, 100, 100), 3)
            # (原圖、圓心、半徑、顏色、粗細)
            # circle outline
            radius = i[2]
            cv.circle(img, center, radius, (255, 255, 255), 3)
            circles_data.append({'number': number, 'center': center, 'r': i})  # 每個圈的數據
            number = number + 1  # 計數器加1
    print(circles_data)
    for item in circles_data:
        text = '%s' % item['number']
        cv.putText(img, text, item['center'], cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2, cv.LINE_AA)
        # 繪製文字(圖片影像/繪製的文字/左上角坐標/字體/字體大小/顏色/字體粗細/字體線條種類)
    cv.imshow("detected circles", src)

    cv.waitKey(0)
    return circles_data


# if __name__ == "__main__":
#     default_file = 'or_1_2.png'
#     src = cv.imread(default_file)  # 讀取圖片
#     main(src)
#     print(circles_data)
#     cv.waitKey(0)0
src = cv.imread('test\\5202-103.png')  # 讀取圖片
# low_green = np.array([0, 0, 90])
# high_green = np.array([255, 255, 255])
# imgHSV = cv.cvtColor(src, cv.COLOR_BGR2HSV)
# mask = cv.inRange(imgHSV, low_green, high_green)
# res = cv.bitwise_and(src, src, mask=mask)
circles_data = main(src)
print(circles_data)