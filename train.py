import argparse
import configparser
import os
import shutil
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from dataset import Train_DatasetLodar, Test_DatasetLodar
from model.restnet_cbam_mixstyle import resnet18
import torch.nn.functional as F



batch_size = 20
epoches = 500

dataset_path = './work_space/data'
train_data = Train_DatasetLodar(dataset_path)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
print('[main]: train_loader is ok')
# test_data = Test_DatasetLodar(dataset_path)
# test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
# print('[main]: test_loader is ok')

model = resnet18(pretrained=False)  # model
model.load_state_dict(torch.load('/home/sunh/github/IndusGrasp/work_space/exp/model_best_399.pt')) ## line pose

if torch.cuda.is_available():
    model.cuda()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('cuda is ok or not:  ', torch.cuda.is_available())


eval_loss = 0
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

    print("learning_rate: ", model.scheduler.get_last_lr())
    ## 50 epoch 保存一次
    if (epoch + 1) % 50 == 0:
        net_dir = os.path.join('./work_space/exp')
        os.makedirs(net_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(net_dir, 'model_best_{}.pt'.format(epoch)))
        print('Saved network')


    # model.eval()
    # total_test_loss = 0
    # with torch.no_grad():
    #     for img,img_edge, pose, path in test_loader:
    #         pose = pose.to(device)
    #         path = path.to(device)
    #         img = img.type(torch.cuda.FloatTensor)
    #         # depths = depths.type(torch.cuda.FloatTensor)
    #         img_fusion = img_fusion.type(torch.cuda.FloatTensor)
    #         img = torch.cat([img, img_fusion], dim=1)
    #
    #         pose_map, path_map = model.forward(img)
    #         loss_pose =  F.smooth_l1_loss(pose_map, pose)
    #         loss_path =  F.smooth_l1_loss(path_map, path)
    #         loss = loss_path + loss_pose
    #         total_test_loss += loss.item()
    # print("Loss on testset: {}".format(total_test_loss))
    #
    # if eval_loss > total_test_loss:
    #     net_dir = os.path.join('./work_space/exp/')
    #     os.makedirs(net_dir, exist_ok=True)
    #     torch.save(model.state_dict(), os.path.join(net_dir, 'model_best_{}.pt'.format(epoch)))
    #     print('Saved network')
    #
    # if epoch==0:
    #     eval_loss = total_test_loss
    # else:
    #     if eval_loss > total_test_loss:
    #         print('total_test_loss down!!!!')
    #         eval_loss = total_test_loss

