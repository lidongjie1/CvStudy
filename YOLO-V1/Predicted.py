import torch
import torchvision
from torchvision import transforms
from torch import nn
import numpy as np
import os
from PIL import Image
import cv2
from matplotlib import pyplot as plt
GL_CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

COLOR = [(255,0,0),(255,125,0),(255,255,0),(255,0,125),(255,0,250),
         (255,125,125),(255,125,250),(125,125,0),(0,255,125),(255,0,0),
         (0,0,255),(125,0,255),(0,125,255),(0,255,255),(125,125,255),
         (0,255,0),(125,255,125),(255,255,255),(100,100,100),(0,0,0),]  # 用来标识20个类别的bbox颜色，可自行设定



def labelbox2(box):
    pass

def draw_box(img,bbox):
    """
    :param img:
    :param bbox: shape(n,6) 0:4 (x1,y1,x2,y2)  5 置信度 6 类别
    :return:
    """
    w,h =img.shape[0:2]
    n = bbox.shape[0]
    for i in range(n):
        confidence = bbox[i, 4]
        if confidence < 0.2:
            continue
        p1 = (int(w * bbox[i, 0]), int(h * bbox[i, 1]))
        p2 = (int(w * bbox[i, 2]), int(h * bbox[i, 3]))
        cls_name = GL_CLASSES[int(bbox[i, 5])]
        print(cls_name, p1, p2)
        cv2.rectangle(img, p1, p2, COLOR[int(bbox[i, 5])])  #
        cv2.putText(img, cls_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.putText(img, str(confidence), (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.imshow("bbox", img)
    cv2.waitKey(0)

# bbox = np.zeros((1,6))
# bbox[0,0] = 165
# bbox[0,1] = 264
# bbox[0,2] = 253
# bbox[0,3] = 372
# bbox[0,4] = 1
# bbox[0,5] = 15
# bbox = torch.tensor(bbox)
# img_dir = r"C:\Users\Adminer\PycharmProjects\ImageProject\data\scripts\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007"
# img_path = os.path.join(img_dir,"JPEGImages")
# img_list = os.listdir(img_path)
# test_img = os.path.join(img_path,img_list[0])
# test_img = cv2.imread(test_img)
# draw_box(test_img,bbox)







img_dir = r"C:\Users\Adminer\PycharmProjects\ImageProject\data\scripts\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\voc2007Yolov1"
img_list = os.listdir(os.path.join(img_dir,"testimg"))
show_dir = os.path.join(img_dir,"testimg")
# img = cv2.imread(show_dir)
# plt.imshow(img)
# plt.show()
trans = transforms.Compose([
    transforms.ToTensor(),
]
)
model = torch.load("YoLoV1_pth")
decice = torch.device("cuda")
for img_name in img_list:
    img_path = os.path.join(show_dir,img_name)
    img = Image.open(img_path).convert("RGB")
    img = trans(img)
    img = img.unsqueeze(0)
    img = img.to(decice)
    preds = torch.squeeze(model(img), dim=0).detach()
    print(preds.shape)
    preds = preds.permute(1,2,0)
    print(preds.shape)
    draw_img = cv2.imwrite(img_path)

