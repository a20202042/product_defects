import serial
from time import sleep
import sys

COM_PORT = 'COM10'  # 請自行修改序列埠名稱
BAUD_RATES = 9600
ser = serial.Serial(COM_PORT, BAUD_RATES)

try:
    while True:
        # 接收用戶的輸入值並轉成小寫
        choice = input('按1亮度提高、  按2亮度下降、  按3最大亮度、  按4最低亮度、  按e關閉程式  ').lower()

        if choice == '1':
            print('提升亮度')
            ser.write(b'HIGH\n')  # 訊息必須是位元組類型
            sleep(0.5)  # 暫停0.5秒，再執行底下接收回應訊息的迴圈
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
            ser.write(b'MINI\n')
            ser.close()
            print('關閉電源')
            sys.exit()
        else:
            print('指令錯誤…')

        while ser.in_waiting:
            mcu_feedback = ser.readline().decode()  # 接收回應訊息並解碼
            print('亮度：', mcu_feedback,'(MAX=225)')

except KeyboardInterrupt:
    ser.close()
    print('再見！')