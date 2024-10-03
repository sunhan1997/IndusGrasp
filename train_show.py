# import matplotlib.pyplot as plt
# import numpy as np
#
# # loss_npy = np.load('/home/sunh/6D_grasp/IndusGrasp/work_space/exp/loss.npy')
# loss_npy = np.load('/home/sunh/6D_grasp/IndusGrasp/work_space/exp/1/test_loss.npy')
# loss_npy2 = np.load('/home/sunh/6D_grasp/IndusGrasp/work_space/exp/2-10-3/test_loss.npy')
# loss_npy3 = np.load('/home/sunh/6D_grasp/IndusGrasp/work_space/exp/3-10-5/test_loss.npy')
# # test_loss_npy = np.load('/home/sunh/6D_grasp/IndusGrasp/work_space/exp/test_loss.npy')
# train_loss_list = []
# train_loss_list2 = []
# train_loss_list3 = []
# # test_loss_list = []
# x_epoch = []
#
# for epoch in range(5,len(loss_npy)):
#
#     loss_one = loss_npy[epoch]
#     # test_loss_one = test_loss_npy[epoch]
#     loss_one2 = loss_npy2[epoch]
#     loss_one3 = loss_npy3[epoch]
#     x_epoch.append(epoch)
#     train_loss_list.append(loss_one)
#     train_loss_list2.append(loss_one2)
#     train_loss_list3.append(loss_one3)
# plt.figure(figsize=(6, 6), dpi=100)
# train_loss_lines = plt.plot(x_epoch, train_loss_list, 'r', lw=1)  # lw为曲线宽度
# val_loss_lines = plt.plot(x_epoch, train_loss_list2, 'b', lw=1)
# val_loss_lines = plt.plot(x_epoch, train_loss_list3, 'g', lw=1)
# plt.title("loss")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.legend(["lr:1e-4","lr:1e-3", "lr:1e-5"])
# # plt.legend(["lr:1e-4  epoch:700"])
# # plt.legend(["train_loss"])
# plt.savefig("val_test2.png")
# plt.show()

pass


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


batch_size = 150
epoches = 400

dataset_path = './work_space/data'
train_data = Train_DatasetLodar(dataset_path)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
print('[main]: train_loader is ok')
test_data = Test_DatasetLodar(dataset_path)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
print('[main]: test_loader is ok')

model = resnet18(pretrained=False)  # model
# model.load_state_dict(torch.load('/home/sunh/6D_grasp/IndusGrasp/work_space/exp/model_best_199.pt')) ## minxtyle

if torch.cuda.is_available():
    model.cuda()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('cuda is ok or not:  ', torch.cuda.is_available())


eval_loss = 0
train_loss_list = []
test_loss_list = []
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
    train_loss_list.append( ( loss_pose.data.float().cpu().numpy() * 3 + loss_path.data.float().cpu().numpy()))



    print("learning_rate: ", model.scheduler.get_last_lr())
    # ## 50 epoch 保存一次
    if (epoch + 1) % 50 == 0:
        net_dir = os.path.join('./work_space/exp')
        os.makedirs(net_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(net_dir, 'model_best_{}.pt'.format(epoch)))
        print('Saved network')

#
#     model.eval()
#     with torch.no_grad():
#         for img,img_fusion, pose, path in test_loader:
#             pose = pose.to(device)
#             path = path.to(device)
#             img = img.type(torch.cuda.FloatTensor)
#             # depths = depths.type(torch.cuda.FloatTensor)
#             img_fusion = img_fusion.type(torch.cuda.FloatTensor)
#             img = torch.cat([img, img_fusion], dim=1)
#
#             pose_map, path_map = model.forward(img)
#             loss_pose =  F.smooth_l1_loss(pose_map, pose)
#             loss_path =  F.smooth_l1_loss(path_map, path)
#     test_loss_list.append(( loss_pose.data.float().cpu().numpy() * 3 + loss_path.data.float().cpu().numpy()))
#
#
#
# np.save('/home/sunh/6D_grasp/IndusGrasp/work_space/exp/loss.npy',train_loss_list)
# np.save('/home/sunh/6D_grasp/IndusGrasp/work_space/exp/test_loss.npy',test_loss_list)

