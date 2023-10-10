import os, random, glob
from shutil import copyfile

imgtxt_dir =r"C:\Users\Adminer\PycharmProjects\ImageProject\data\scripts\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\voc2007Yolov1\test.txt"
img_dir = r'C:\Users\Adminer\PycharmProjects\ImageProject\data\scripts\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\voc2007Yolov1\img'
save_testimg = r'C:\Users\Adminer\PycharmProjects\ImageProject\data\scripts\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\voc2007Yolov1\testimg'
choseImg = []


#通过glob.glob来获取原始路径下，所有'.jpg'文件
imageList1 = glob.glob(os.path.join(img_dir, '*.jpg'))

f = open(imgtxt_dir,"r")   #设置文件对象
line = f.readline()
line = line[:-1]
while line:             #直到读取完文件
    line = f.readline().strip() #读取一行文件，包括换行符
    if os.path.exists(line):
        choseImg.append(os.path.basename(line))

for i in choseImg:
    # 将随机选中的jpg文件遍历复制到目标文件夹中
    copyfile(img_dir + '/' + i, save_testimg + '/' + i)

f.close() #关闭文件
