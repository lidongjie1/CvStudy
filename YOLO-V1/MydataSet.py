from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import torch
from PIL import Image
import torchvision.transforms as transforms

class MydataSet(Dataset):
    def __init__(self,dataset_dir, seed=None, mode="train",trans=None):
        if seed==None:
            seed = random.randint(0,65536)
        random.seed(seed)
        self.dataset_dir = dataset_dir
        self.mode = mode
        img_list_txt = os.path.join(self.dataset_dir,self.mode+".txt")
        label_csv= os.path.join(self.dataset_dir,self.mode+".csv")
        self.img_list = []
        self.label = np.loadtxt(label_csv)  # 读取标签数组文件
        # 读取图片位置文件
        with open(img_list_txt, 'r') as f:
            for line in f.readlines():
                self.img_list.append(line.strip())
        # 在mode=train或val时， 将数据进行切分
        # 注意在mode="val"时，传入的随机种子seed要和mode="train"相同
        self.num_all_data = len(self.img_list)
        num_train = int(self.num_all_data)
        self.all_ids = list(range(self.num_all_data))
        self.all_ids = self.all_ids[:num_train]
        self.trans = trans


    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, item):
        id = self.all_ids[item]
        label = torch.tensor(self.label[id, :])
        img_path = self.img_list[id]
        img = Image.open(img_path)
        if self.trans is None:
            trans = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            trans = self.trans
        img = trans(img)  # 图像预处理&数据增广
        # transforms.ToPILImage()(img).show()  # for debug
        # print(label)
        return img, label



        
if __name__ == '__main__':
    # 调试用，依次取出数据看看是否正确
    dataset_dir = r"C:\Users\Adminer\PycharmProjects\ImageProject\data\scripts\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\voc2007Yolov1"
    dataset = MydataSet(dataset_dir)
    dataloader = DataLoader(dataset, 1)
    for i in enumerate(dataloader):
        input("press enter to continue")
