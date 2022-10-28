import serial
from time import sleep
import sys
COM_PORT = 'COM10'  # 請自行修改序列埠名稱
BAUD_RATES = 9600
ser = serial.Serial(COM_PORT, BAUD_RATES)
def change_color():
    choice = input('r 紅光、  g 綠光、  b 藍光、  w 白光、  按e關閉程式  ').lower()
    if choice == 'r':
        print('紅光')
        ser.write(b'RED\n')  # 訊息必須是位元組類型
        sleep(0.5)  # 暫停0.5秒，再執行底下接收回應訊息的迴圈
    elif choice == 'g':
        print('綠光')
        ser.write(b'GREEN\n')
        sleep(0.5)

    elif choice == 'b':
        print('藍光')
        ser.write(b'BLUE\n')
        sleep(0.5)

    elif choice == 'w':
        print('白光')
        ser.write(b'WHITE\n')
        sleep(0.5)

    elif choice == 'e':
        ser.write(b'CLOSE\n')
        ser.close()
        print('關閉電源')
        sys.exit()
    else:
        print('指令錯誤…')


try:
    while True:
        choice = input('按0調整顏色、  按1亮度提高、  按2亮度下降、  按3最大亮度、  按4最低亮度、  按e關閉程式  ').lower()
        if choice == '0':
            print('顏色調整')
            ser.write(b'COLOR\n')
            sleep(0.5)
            change_color()
        elif choice == '1':
            print('提高亮度')
            ser.write(b'HIGH\n')
            sleep(0.5)
        elif choice == '2':
            print('降低亮度')
            ser.write(b'LOW\n')
            sleep(0.5)
        elif choice == '3':
            print('最大亮度')
            ser.write(b'MAX\n')
            sleep(0.5)
        elif choice == '4':
            print('最低亮度')
            ser.write(b'MINI\n')
            sleep(0.5)
        elif choice == 'e':
            ser.write(b'CLOSE\n')
            ser.close()
            print('關閉電源')
            sys.exit()
        else:
            print('指令錯誤…')
        while ser.in_waiting:
            mcu_feedback = ser.readline().decode()  # 接收回應訊息並解碼
            print('亮度：', mcu_feedback, '(MAX=225)')


except KeyboardInterrupt:
    ser.close()
    print('再見！')