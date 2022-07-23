import os, cv2
dir = 'match_data'
files = os.listdir(dir)
file_data = {}
for file in files:
    template = cv2.imread(dir + '\\' + file)
    file_data.update({str(file): template})
# print(file_data)
print(file_data.get('or_3.png'))
a = ['or_3.png', 'or_6.png']
# for item in a:
#     print(a)
#     print(file_data.items(item))