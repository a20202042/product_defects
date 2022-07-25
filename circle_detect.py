import sys
import cv2 as cv
import numpy as np

#vvvvvv
def main(argv):
    default_file = '0.png'
    src = cv.imread(default_file) #讀取圖片
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY) #轉灰階

    gray = cv.medianBlur(gray, 5)

    rows = gray.shape[0]
    circles = cv.HoughCircles(gray,
                              cv.HOUGH_GRADIENT, #
                              minDist=6,
                              #圓心距離
                              dp=1,
                              #檢測圓心的累加器精度和圖像精度比的倒數(1=相同分辨綠，2=累加器是輸入圖案一半大的寬高)
                              param1=150,
                              #canny檢測的高闊值，低闊值為一半
                              param2=15,
                              #圓心的累加器闊值，越小檢測更多的圓，越大越精確
                              minRadius=8,
                              #最小半徑
                              maxRadius=10)
                              #最大半徑
    circles_data = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        #四捨五入(np.around)後，轉換成16位無符號整數(np.uint16)。
        number = 1
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            #(原圖、圓心、半徑、顏色、粗細)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
            circles_data.append({'number': number, 'center': center, 'r': i}) #每個圈的數據
            number = number + 1 #計數器加1
    print(circles_data)
    for item in circles_data:
        text = 'circle_%s' % item['number']
        cv.putText(src, text, item['center'], cv.FONT_HERSHEY_SIMPLEX,
                   1, (255, 255, 0), 2, cv.LINE_AA)
        #繪製文字(圖片影像/繪製的文字/左上角坐標/字體/字體大小/顏色/字體粗細/字體線條種類)
    cv.imshow("detected circles", src)

    cv.waitKey(0)
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
