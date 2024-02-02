# import numpy as np
# import cv2
# import torch
#
# from dataset import Dataset
# import resnet50_kps4_add_SE_CBAM as net
# import time
# import torchvision.transforms.functional as F
# from utils.kps_heatmap import preprocess_rgb,heatmap_to_point
# import matplotlib.pyplot as plt
# from skimage.filters import gaussian
# from skimage.feature import peak_local_max
#
# import utils.eval_grasp_util as util
# import open3d as o3d
# from utils.collision_detector import ModelFreeCollisionDetector
# from graspnetAPI import GraspGroup
# from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
# import math
#
# def collision_detection(gg, cloud):
#     mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=0.01)
#     collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
#     gg = gg[~collision_mask]
#     return gg
#
# def vis_grasps(gg, cloud):
#     gg.nms()
#     gg.sort_by_score()
#     gg = gg[:50]
#     grippers = gg.to_open3d_geometry_list()
#     o3d.visualization.draw_geometries([cloud, *grippers])
#
# ##############################################  创建数据集类  ########################################
# dataset_path = '/home/sunh/6D_ws/MPGrasp/work_space/data'
# dataset = Dataset(dataset_path)
# # dataset.get_training_images(dataset_path)  # get train image
# ##############################################  dataset_load  ########################################
# # load model
# model = net.resnet18()  # model
# # model.load_state_dict(torch.load('/home/sunh/6D_ws/MPGrasp/work_space/exp/grasp_success/model_best_192.pt'))
# # model.load_state_dict(torch.load('/home/sunh/6D_ws/MPGrasp/work_space/exp/without_bg_one_channel_network/model_best_89.pt')) ## line
# # model.load_state_dict(torch.load('/home/sunh/6D_ws/MPGrasp/work_space/exp/验证CBAM/circle_cbam/2.without_bg_one_channel_circle/model_best_399.pt'))  ## circle
# # model.load_state_dict(torch.load('/home/sunh/6D_ws/MPGrasp/work_space/exp/验证CBAM/santong_se/model_best_49.pt'))  ## san tong
# # model.load_state_dict(torch.load('/home/sunh/6D_ws/MPGrasp/work_space/exp/验证Fusion/line_rgb/model_best_29.pt'))  ## san tong
#
#
# model.load_state_dict(torch.load('/home/sunh/6D_ws/MPGrasp/work_space/exp/1.for_picture/model_best_59.pt')) ## line
#
# print('cuda is ok or not:  ', torch.cuda.is_available())
# model.cuda()
# model.eval()
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# #  载入fast rcnn训练结果 这里使用的是torchvision的
# fast_rcnn = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/omron/model_50_2.pth')
# # fast_rcnn = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/grasp2/model_50.pth')
# # fast_rcnn = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/grasp3/model_50.pth')
# # fast_rcnn = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/grasp4/model_50.pth')
# # fast_rcnn = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/grasp5/model_50.pth')
# # fast_rcnn = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/grasp6/model_50.pth')
# fast_rcnn.to(device)
# fast_rcnn.eval()
#
#
# with torch.no_grad():
#     for idx in range( 100, 1400):
#         # img,fusion, gt_pose, path = dataset.torch_batch([idx])
#         ## 读取图像
#         img_o = cv2.imread('/home/sunh/6D_ws/my_dataset/ObjectDatasetTools-master/LINEMOD/gr1/JPEGImages/{}.jpg'.format(idx))
#         depth = cv2.imread('/home/sunh/6D_ws/my_dataset/ObjectDatasetTools-master/LINEMOD/gr1/depth/{}.png'.format(idx), cv2.IMREAD_ANYDEPTH)
#         # img_o = cv2.imread('/home/sunh/6D_ws/my_dataset/ObjectDatasetTools-master/LINEMOD/test_grasp/3.jpg')
#         start = time.time()
#
#         rgb = img_o.copy()
#         img = img_o.copy()
#
#         ################ fast rcnn ######################
#         # rgb = torch.from_numpy(rgb.transpose((2, 0, 1)))
#         # rgb = rgb.float().div(255)
#         # with torch.no_grad():
#         #     prediction = fast_rcnn([rgb.to(device)])
#         #     boxes = prediction[0]['boxes']
#         #     scores = prediction[0]['scores']
#         #     scores_np = scores.cpu().numpy()
#         #     if len(scores_np) == 0:
#         #         continue
#         #     scores_max = scores_np.max()
#         #     if scores_max< 0.5:
#         #         continue
#         #     for idx in range(boxes.shape[0]):
#         #         first_score = scores[0]
#         #         if scores[idx] >= first_score:
#         #             first_score = scores[idx]
#         #             # x1, y1, x2, y2 = int(boxes[idx][0])-4, int(boxes[idx][1])-4, int(boxes[idx][2])+4, int(boxes[idx][3])+4
#         #             # x1, y1, x2, y2 = int(boxes[idx][0]) - 2, int(boxes[idx][1]) - 2, int(boxes[idx][2]) + 4, int(boxes[idx][3]) + 4
#         #             x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
#         #     rgb = img[int(y1):int(y2), int(x1): int(x2)]
#         #     w, h = int(x2) - int(x1 ), int(y2)- int(y1)
#
#
#         rgb = torch.from_numpy(rgb.transpose((2, 0, 1)))
#         rgb = rgb.float().div(255)
#         with torch.no_grad():
#             prediction = fast_rcnn([rgb.to(device)])
#             boxes = prediction[0]['boxes']
#             scores = prediction[0]['scores']
#
#         all_point = []
#         all_bbox = []
#         for idx in range(boxes.shape[0]):
#             if scores[idx] >= 0.7:
#                 # x1, y1, x2, y2 = int(boxes[idx][0])-4, int(boxes[idx][1])-4, int(boxes[idx][2])+8, int(boxes[idx][3])+8
#                 x1, y1, x2, y2 = int(boxes[idx][0])-2, int(boxes[idx][1])-2, int(boxes[idx][2])+4, int(boxes[idx][3])+4
#                 x, y = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
#                 cv2.circle(img_o, (x, y), 3, (255, 0, 0), 2)  # 抓取关键点1
#                 cv2.circle(img_o, (x, y), 40, (0, 255, 0), 2)  # 抓取关键点1
#                 all_point.append((x, y))
#                 all_bbox.append((x1, y1, x2, y2))
#                 # cv2.rectangle(img_show, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=2)
#
#         all_score = []
#         all_dis = []
#         for idx in range(len(all_point)):
#             x, y = all_point[idx]
#             score = 0
#             dis_m = 0
#
#             for i in range(len(all_point)):
#                 if i == idx:
#                     continue
#                 x_other, y_other = all_point[i]
#                 dis = np.sqrt((x - x_other) * (x - x_other) + (y - y_other) * (y - y_other))
#                 if dis < 50:
#                     dis_m += dis
#                     score += 1
#
#             all_dis.append(dis_m)
#             all_score.append(score)
#
#         all_score = np.array(all_score)
#         index_s = np.argmin(all_score)
#         all_dis = np.array(all_dis)
#         index_d = np.argmin(all_dis)
#         index = index_d
#         cv2.circle(img_o, (all_point[index][0], all_point[index][1]), 3, (0, 0, 255), 2)  # 抓取关键点1
#         x1, y1, x2, y2 = all_bbox[index]
#         rgb = img[int(y1):int(y2), int(x1): int(x2)]
#         w, h = int(x2) - int(x1), int(y2) - int(y1),
#         print('index_s:   ', index_s)
#         print('index_d:   ', index_d)
#         # cv2.imshow('s',rgb)
#         # cv2.waitKey()
#         ################ fast rcnn ######################
#
#
#
#
#         ##### 零件剪裁出来并resize到128，然后rgb为img_x，多特征图为img_fusion
#         img = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_NEAREST)
#         img_fusion = dataset.gen_multi_feature_fusion_map(img)
#
#         img_x = F.to_pil_image(img.astype(np.uint8))
#         img_x = preprocess_rgb(img_x)
#         img_fusion = F.to_pil_image(img_fusion.astype(np.uint8))
#         img_fusion = preprocess_rgb(img_fusion)
#
#         img_x = img_x.unsqueeze(0)
#         img_x = img_x.type(torch.cuda.FloatTensor)
#         img_fusion = img_fusion.unsqueeze(0)
#         img_fusion = img_fusion.type(torch.cuda.FloatTensor)
#         # 把img_x和img_fusion 做一个cat
#         img_x = torch.cat([img_x, img_fusion], dim=1)
#         # 网络推理
#         model.forward(img_x)
#
#         ##### 读取结果
#         pre_pose,pre_path = model.read_network_output()
#         pre_pose = gaussian(pre_pose, 2.0, preserve_range=True)
#         pre_path = gaussian(pre_path, 2.0, preserve_range=True)
#
#
#         local_max_pose = peak_local_max(pre_pose, min_distance=20, threshold_abs=0.2, num_peaks=2) # 在抓取关键点图中找到两个局部最大点
#         local_max_path = peak_local_max(pre_path, min_distance=20, threshold_abs=0.2, num_peaks=1) # 在抓取路径图中找到一个局部最大点作为抓取中心
#         g_point = []
#         for grasp_point_array in local_max_pose:
#             grasp_point = tuple(grasp_point_array)
#             g_point.append(grasp_point)
#
#         if len(g_point) != 2:
#             continue
#         grasp_kps_map = np.zeros((128,128,3))
#         cv2.circle(grasp_kps_map, (g_point[0][1],g_point[0][0]), 3, (255,0,0), 2)    # 抓取关键点1
#         cv2.circle(grasp_kps_map, (g_point[1][1],g_point[1][0]), 3, (0,255, 0), 2)  # 抓取关键点2
#         # cv2.circle(grasp_kps_map, (local_max_path[0][1],local_max_path[0][0]), 2, (0,0, 255), 2)  # 抓取关键点center
#         nn = int ((g_point[0][1]+ g_point[1][1])/2.0)
#         mm = int ((g_point[0][0]+ g_point[1][0])/2.0)
#         cv2.circle(grasp_kps_map, (nn,mm), 2, (0,0, 255), 2)  # 抓取关键点center
#
#
#         ## get grasp score
#         # score1, score2 = util.eval_grasp(local_max_path, g_point)
#         # print(score1,score2)
#         # if score1 >0.8 or score2>0.8:
#         #     continue
#
#         ## get grasp key point and center
#         grasp_kps_1, grasp_kps_2, grasp_center = util.process_grasp_kps_center(w,h,grasp_kps_map)
#         cv2.circle(img_o, (grasp_kps_1[0] + x1, grasp_kps_1[1] + y1), 3, (255, 0, 0), 2)
#         cv2.circle(img_o, (grasp_kps_2[0] + x1, grasp_kps_2[1] + y1), 3, (0, 255, 0), 2)
#         cv2.imshow('img_o', img_o)
#         # cv2.imshow('grasp_kps_map', grasp_kps_map)
#         cv2.imshow('pre_pose', pre_pose)
#         cv2.imshow('pre_path', pre_path)
#         cv2.imshow('rgb', rgb)
#         cv2.waitKey(0)
#
#         # ## get grasp points
#         grasp_points = util.get_grasp_points(grasp_kps_1,grasp_kps_2)
#
#         ## 计算抓取关键点连线与水平线夹角，作为抓取角度
#         dx = grasp_kps_2[0] - grasp_kps_1[0]
#         dy = grasp_kps_2[1] - grasp_kps_1[1]
#         angle = (np.arctan2(-dy, dx) + np.pi / 2) % np.pi - np.pi / 2
#
#         img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)  # 图像变回原来尺寸
#         T = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0],[0, 0, 0, 1]])
#         gg_list = []
#         for i in range(len(grasp_points)):
#             cv2.circle(img, (int(grasp_points[i][0]), int(grasp_points[i][1])), 2, (0, 255, 255), 2)
#             point_depth = depth[int(grasp_center[1] + y1), int(grasp_center[0] + x1)] * 0.00012498664727900177
#             translation = util.uv2xyz((int(grasp_points[i][0]+ x1), int(grasp_points[i][1]+y1)), point_depth) + np.array([0,0,0.005])
#
#             g_T = np.identity(4)
#             g_T[0:3, 3] = translation
#             theta = np.array([0, 0, (3.1415926 / 2) - angle])
#             g_T[:3, :3] = util.angle2Rmat(theta)
#             g_T = np.dot(g_T, T)
#             g = np.array([0.8, 0.015, 0.2, 0.0,  ## score with hight detpt
#                           g_T[0, 0], g_T[0, 1], g_T[0, 2], g_T[1, 0],
#                           g_T[1, 1], g_T[1, 2], g_T[2, 0], g_T[2, 1],
#                           g_T[2, 2], g_T[0, 3], g_T[1, 3], g_T[2, 3], -1.00000000e+00])
#             gg_list.append(g)
#
#             # ### get grasp with and fast collision detection
#             # griper1, griper2 = util.get_grasp_with(pre_pose, w, h, grasp_points[i], grasp_kps_1, grasp_kps_2)
#             # collision_result = util.fast_collision_detection(grasp_center, griper1, griper2, depth,x1,y1)
#             # if collision_result:
#             #     cv2.circle(img, (griper1[0], griper1[1]), 3, (255, 0, 0), 2)
#             #     cv2.circle(img, (griper2[0], griper2[1]), 3, (0, 255, 0), 2)
#
#         cv2.imshow('img', img)
#         cv2.waitKey(0)
#
#         # color = img_o.copy()
#         # color = cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
#         # color = color/255.0
#         # workspace_mask = np.ones((480, 640), dtype=bool)
#         # workspace_mask[0:100,: ] = False
#         # workspace_mask[:,0:100 ] = False
#         # workspace_mask[400:480, :] = False
#         # workspace_mask[:, 500:640] = False
#         # camera = CameraInfo(640.0, 480.0, 621.4, 621.4, 313.721, 239.976, float(1 / 0.00012498664727900177))
#         # cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
#         # mask = (workspace_mask & (depth > 0))
#         # cloud_masked = cloud[mask]
#         # color_masked = color[mask]
#         # cloud = o3d.geometry.PointCloud()
#         # cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
#         # cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
#         #
#         # cl, ind = cloud.remove_statistical_outlier(nb_neighbors=70, std_ratio=1)
#         # # cl, ind = cloud.remove_radius_outlier(nb_points=50, radius=2)
#         # cloud = cloud.select_by_index(ind)
#         #
#         # gg_array = np.array(gg_list)
#         # gg = GraspGroup(gg_array)
#         # vis_grasps(gg, cloud)
#         #
#         # gg = collision_detection(gg, np.array(cloud.points))
#         # vis_grasps(gg, cloud)




import numpy as np
import cv2
import torch
from dataset import gen_multi_feature_fusion_map
import time
import torchvision.transforms.functional as F
from utils.kps_heatmap import preprocess_rgb
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.feature import peak_local_max

import utils.eval_grasp_util as util
import open3d as o3d
from utils.collision_detector import ModelFreeCollisionDetector
from graspnetAPI import GraspGroup
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
import math
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


##############################################  dataset_load  ########################################
# load model
model = resnet18()  # model
# model.load_state_dict(torch.load('/home/sunh/github/IndusGrasp/work_space/exp/model_best_9.pt')) ## minxtyle
model.load_state_dict(torch.load('/home/sunh/github/IndusGrasp/work_space/exp/key_path/model_best_14.pt'))
print('cuda is ok or not:  ', torch.cuda.is_available())
model.cuda()
model.eval()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#  载入fast rcnn训练结果 这里使用的是torchvision的
fast_rcnn = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/omron/model_50_2.pth') # 207
# fast_rcnn = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/grasp2/model_50.pth') ##49 50 233
# fast_rcnn = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/grasp3/model_50.pth') ## 215  gr32:41  all_bbox[0]    ### gr3  : 215 zhe dang path
# fast_rcnn = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/grasp4/model_50.pth')
# fast_rcnn = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/grasp5/model_50.pth')
# fast_rcnn = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/grasp6/model_50.pth') # gr62： 0,58 94
fast_rcnn.to(device)
fast_rcnn.eval()



with torch.no_grad():
    for idx in range( 9, 1400): #
        print('-----------: ', idx)
        # rgb,fusion, gt_pose, path = dataset.torch_batch([idx])
        ## 读取图像
        img_o = cv2.imread('/media/sunh/Samsung_T5/6D_data/my_Dataset/ObjectDatasetTools-master/LINEMOD2/gr1/JPEGImages/{}.jpg'.format(idx))
        depth = cv2.imread('/media/sunh/Samsung_T5/6D_data/my_Dataset/ObjectDatasetTools-master/LINEMOD2/gr1/depth/{}.png'.format(idx), cv2.IMREAD_ANYDEPTH)
        start = time.time()

        rgb = img_o.copy()
        img = img_o.copy()
        ################ fast rcnn ######################
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1)))
        rgb = rgb.float().div(255)
        with torch.no_grad():
            prediction = fast_rcnn([rgb.to(device)])
            boxes = prediction[0]['boxes']
            scores = prediction[0]['scores']

        all_point = []
        all_bbox = []
        for idx in range(boxes.shape[0]):
            if scores[idx] >= 0.7:
                # x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
                x1, y1, x2, y2 = int(boxes[idx][0])-2, int(boxes[idx][1])-2, int(boxes[idx][2])+4, int(boxes[idx][3])+4
                x, y = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
                all_point.append((x, y))
                all_bbox.append((x1, y1, x2, y2))
                # cv2.rectangle(img_o, (x1,y1),(x2,y2), (0, 255, 0), 2)
                # cv2.imshow('img_o', img_o)
                # cv2.waitKey()

        all_score = []
        all_dis = []
        for idx in range(len(all_point)):
            x, y = all_point[idx]
            score = 0
            dis_m = 0

            for i in range(len(all_point)):
                if i == idx:
                    continue
                x_other, y_other = all_point[i]
                dis = np.sqrt((x - x_other) * (x - x_other) + (y - y_other) * (y - y_other))
                if dis < 20:
                    dis_m += dis
                    score += 1

            all_dis.append(dis_m)
            all_score.append(score)

        all_score = np.array(all_score)
        index_s = np.argmin(all_score)
        all_dis = np.array(all_dis)
        index_d = np.argmin(all_dis)
        index = index_d
        # cv2.circle(img_o, (all_point[index][0], all_point[index][1]), 3, (0, 0, 255), 2)
        x1, y1, x2, y2 = all_bbox[index]
        # x1, y1, x2, y2 = all_bbox[2]
        # cv2.rectangle(img_o, (x1,y1),(x2,y2), (0, 255, 0), 2)

        rgb = img[int(y1):int(y2), int(x1): int(x2)]
        rgb2 = img_o[int(y1):int(y2), int(x1): int(x2)]  ## circle
        w, h = int(x2) - int(x1), int(y2) - int(y1),
        print('all_bbox[index]:   ', all_bbox[index])
        ################ fast rcnn ######################


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


        local_max_pose = peak_local_max(pre_path, min_distance=10, threshold_abs=0.2,
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

        if len(gk_point) != 2 :
            print('wrong ****************')
            continue

        grasp_kps_map = np.zeros((128, 128, 3))
        cv2.circle(grasp_kps_map, (gk_point[0][1], gk_point[0][0]), 2, (255, 0, 0), 2)  # 抓取关键点1
        cv2.circle(grasp_kps_map, (gk_point[1][1], gk_point[1][0]), 2, (0, 255, 0), 2)  # 抓取关键点2
        cv2.circle(grasp_kps_map, (g_point[0][1], g_point[0][0]), 5, (0, 0, 255), 2)  # 抓取关键点2
        grasp_kps_map = cv2.resize(grasp_kps_map, (w, h), interpolation=cv2.INTER_NEAREST)  # 图像变回原来尺寸


        grasp_kps_1, grasp_kps_2, grasp_center = util.process_grasp_kps_center_circle(w, h, grasp_kps_map, rgb2)

        ## 计算抓取关键点连线与水平线夹角，作为抓取角度
        dx = grasp_kps_1[0] - grasp_kps_2[0]
        dy = grasp_kps_1[1] - grasp_kps_2[1]
        angle_grasp = (np.arctan2(-dy, dx) + np.pi / 2) % np.pi - np.pi / 2


        depth_x = grasp_kps_1[0]+x1
        depth_y =  grasp_kps_1[1]+y1
        cv2.circle(img_o, (grasp_kps_1[0]+x1, grasp_kps_1[1]+y1), 3, (255, 0, 0), 2)
        cv2.circle(img_o, (grasp_kps_2[0]+x1, grasp_kps_2[1]+y1), 3, (0, 255, 0), 2)
        cv2.circle(img_o, (grasp_center[0]+x1, grasp_center[1]+y1), 3, (0, 0, 255), 2)

        point_depth = depth[depth_y, depth_x]

        cv2.imshow('pre_p', pre_pose)
        cv2.imshow('pre_path', pre_path)
        cv2.imshow('img_o', img_o)
        cv2.imshow('rgb', rgb)

        # cv2.imwrite('/home/sunh/Pictures/MPgrasp2/path_effect/output_occ.png',img_o)
        # cv2.imwrite('/home/sunh/Pictures/MPgrasp2/path_effect/rgb_with_path.png',rgb)

        cv2.waitKey(0)

        # plt.imshow(pre_pose)
        # plt.colorbar()
        # plt.show()
        # plt.imshow(pre_path)
        # plt.colorbar()
        # plt.show()


