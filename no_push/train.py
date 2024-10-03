import argparse
import configparser
import os
import shutil
import numpy as np
import cv2
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import progressbar

from dataset import Dataset
import torchvision.transforms.functional as F
from utils.kps_heatmap import preprocess_rgb

from model.restnet_cbam_mixstyle import resnet18

##############################################  创建数据集  ########################################
dataset_path = './work_space/data'
dataset = Dataset(dataset_path)
dataset.load_bg_images_sunhan(dataset_path)   # get bg image
dataset.get_training_images(dataset_path)  # get train image
vis = False
first  = False
##############################################  创建数据集  ########################################

##############################################  网络设置  ########################################
batch_size = 20
lr = 1e-5
weight_decay = 3e-5
epoches = 500
##############################################  网络设置  ########################################


class MyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self,index):
        img,img_fusion, pose, path  = self.dataset.torch_batch([index])
        img = F.to_pil_image(img.astype(np.uint8))
        img = preprocess_rgb(img)
        img_fusion = F.to_pil_image(img_fusion.astype(np.uint8))
        img_fusion = preprocess_rgb(img_fusion)

        pose = torch.tensor(pose, dtype=torch.float32)
        path = torch.tensor(path, dtype=torch.float32)
        return img,img_fusion,pose, path

    def __len__(self):
        return self.dataset.imgs_numb_all

# ##################################### do_train ##############################
# load data
train_data = MyDataset(dataset=dataset, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
print('[main]: train_loader is ok')

model = resnet18(pretrained=False)  # model
model.load_state_dict(torch.load('/home/sunh/github/IndusGrasp/work_space/exp/model_best_139.pt'),strict=False ) # 载入权重  strict=False
if torch.cuda.is_available():
    model.cuda()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('cuda is ok or not:  ', torch.cuda.is_available())



for epoch in range(epoches):

    for img, img_fusion, pose, path  in train_loader:
        pose = pose.to(device)
        path = path.to(device)
        img = img.type(torch.cuda.FloatTensor)
        # depths = depths.type(torch.cuda.FloatTensor)
        img_fusion = img_fusion.type(torch.cuda.FloatTensor)
        img = torch.cat([img, img_fusion], dim=1)

         ## 开始训练并输出loss
        loss_pose,loss_path = model.optimize(img, pose,path  )
        print("epoch: {} >>> loss_p: {} >>> loss_c: {}  ".format(epoch, loss_pose.data.float() * 100, loss_path.data.float() * 100  ))

    print("epoch: {} >>> loss_p: {} >>> loss_c: {}  ".format(epoch, loss_pose.data.float() * 100, loss_path.data.float() * 100 ))

    ## 50 epoch 保存一次
    if (epoch + 1) % 20 == 0:
        net_dir = os.path.join('./work_space/exp')
        os.makedirs(net_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(net_dir, 'model_best_{}.pt'.format(epoch)))
        print('Saved network')


