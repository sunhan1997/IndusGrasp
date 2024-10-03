import numpy as np
import cv2
import torch
import time
import torchvision.transforms.functional as F
from utils.kps_heatmap import preprocess_rgb
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.feature import peak_local_max

import open3d as o3d
from utils.collision_detector import ModelFreeCollisionDetector

from model.restnet_cbam_mixstyle import resnet18


def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=0.01)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

# 直线检测
def fld(img,test_model=False):
    H, W = img.shape[0],img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fld = cv2.ximgproc.createFastLineDetector()
    dlines = fld.detect(gray)

    if test_model:
        show = np.zeros([H, W])
    else:
        show = np.zeros([H, W])

    if dlines is None:
        img_edge = cv2.Canny(img, 50, 200)
        return img_edge
    else:
        for dline in dlines:
            x0 = int(round(dline[0][0]))
            y0 = int(round(dline[0][1]))
            x1 = int(round(dline[0][2]))
            y1 = int(round(dline[0][3]))
            cv2.line(show, (x0, y0), (x1, y1), (255, 255, 255), 1, cv2.LINE_AA)
        return show

def gen_multi_feature_fusion_map(img, test_model=False):
    img_fld = fld(img.copy(), test_model)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edge = cv2.Canny(img, 50, 200)

    img_gray = np.expand_dims(img_gray, axis=2)
    img_fld = np.expand_dims(img_fld, axis=2)
    img_edge = np.expand_dims(img_edge, axis=2)

    img_add = np.concatenate([img_gray, img_edge], 2)
    img_add = np.concatenate([img_add, img_fld], 2)

    return img_add

##############################################  dataset_load  ########################################
# load model
model = resnet18()  # model
mmm = 5

model.load_state_dict(torch.load('./work_space/exp/model_best_9.pt')) ##   1e-4

print('cuda is ok or not:  ', torch.cuda.is_available())
model.cuda()
model.eval()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#  载入fast rcnn训练结果 这里使用的是torchvision的
# fast_rcnn = torch.load('/home/sunh/6D_grasp/IndusGrasp/maskrcnn/trained/model_50.pth') # 207
# fast_rcnn.to(device)
# fast_rcnn.eval()

with torch.no_grad():
    for idx in range(85,89):
        print(idx)
        ## 读取图像
        img_o = cv2.imread('./example/singel_obj/{}.jpg'.format(idx))
        start = time.time()

        rgb = img_o.copy()
        rgb2 = img_o.copy()


        ##### 零件剪裁出来并resize到128，然后rgb为img_x，多特征图为img_fusion
        rgb = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_NEAREST)
        img_fusion = gen_multi_feature_fusion_map(rgb)

        img_x = F.to_pil_image(rgb.astype(np.uint8))
        img_x = preprocess_rgb(img_x)
        img_fusion = F.to_pil_image(img_fusion.astype(np.uint8))
        img_fusion = preprocess_rgb(img_fusion)

        img_x = img_x.unsqueeze(0)
        img_x = img_x.type(torch.cuda.FloatTensor)
        img_fusion = img_fusion.unsqueeze(0)
        img_fusion = img_fusion.type(torch.cuda.FloatTensor)
        # 把img_x和img_fusion 做一个cat
        img_x = torch.cat([img_x, img_fusion], dim=1)
        # 网络推理
        model.forward(img_x)

        ##### 读取结果
        pre_pose,pre_path = model.read_network_output()
        pre_pose = gaussian(pre_pose, 2.0, preserve_range=True)
        pre_path = gaussian(pre_path, 2.0, preserve_range=True)

        plt.imshow(pre_pose)
        plt.colorbar()
        plt.show()
        plt.imshow(pre_path)
        plt.colorbar()
        plt.show()

        local_max_pose = peak_local_max(pre_pose, min_distance=10, threshold_abs=0.2,
                                                num_peaks=2)  # 在抓取关键点图中找到两个局部最大点
        local_max_path = peak_local_max(pre_path, min_distance=10, threshold_abs=0.2,
                                                num_peaks=1)  # 在抓取关键点图中找到两个局部最大点
        gk_point = []
        for grasp_point_array in local_max_pose:
            grasp_point = tuple(grasp_point_array)
            gk_point.append(grasp_point)

        g_point = []
        for grasp_point_array in local_max_path:
            grasp_point = tuple(grasp_point_array)
            g_point.append(grasp_point)

        grasp_kps_map = np.zeros((128, 128, 3))
        cv2.circle(rgb, (gk_point[0][1], gk_point[0][0]), 3, (255, 0, 0), 2)  # 抓取关键点1
        cv2.circle(rgb, (gk_point[1][1], gk_point[1][0]), 3, (0, 255, 0), 2)  # 抓取关键点2
        cv2.circle(rgb, (g_point[0][1], g_point[0][0]), 3, (0, 0, 255), 2)  # 抓取关键点2

        cv2.imshow('rgb', rgb)
        cv2.waitKey(0)





