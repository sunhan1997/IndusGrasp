#!/home/sunh/anaconda3/envs/yolo/bin/python
# -*- coding:utf8 -*-
from sys import path
path.insert(0,'/home/sunh/cv_bridge_ws/devel/lib/python3/dist-packages')
import sys
sys.path.append("/opt/ros/melodic/lib/python2.7/dist-packages")

import numpy as np
import cv2
import torch

from dataset import gen_multi_feature_fusion_map
# import resnet50_kps4_add_SE_CBAM as net
from model.restnet_cbam_mixstyle import resnet18
import time

import torchvision.transforms.functional as F
from utils.kps_heatmap import preprocess_rgb,heatmap_to_point
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.feature import peak_local_max

import threading
import rospy
from std_msgs.msg import Float64MultiArray
import message_filters
from message_filters import ApproximateTimeSynchronizer
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
import pyrealsense2 as rs
import open3d as o3d
import cv2
import utils.eval_grasp_util as util



#相机配置
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

#相机深度参数，包括精度以及 depth_scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 3)
depth_scale = depth_sensor.get_depth_scale()
clipping_distance_in_meters = 8  # 8 meter
clipping_distance = clipping_distance_in_meters / depth_scale
#color和depth对齐
align_to = rs.stream.color
align = rs.align(align_to)

## 2维点转换为三维点
def uv2xyz(uv,z,cx = 323.844,cy =232.171,fx=615.372,fy=615.312):
# def uv2xyz(uv,z,cx = 326.60858154296875,cy = 245.27972412109375,fx=603.3093872070312,fy=602.8275756835938): ### pjlab
    xcoord = (uv[0] - cx) * z / fx
    ycoord = (uv[1]- cy) * z/ fy
    return xcoord,ycoord,z



class DetectNode:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.net = net.resnet18()  # grasp model
        self.net = resnet18()  # grasp model
        self.net.load_state_dict(
            torch.load('./work_space/exp/model_best_9.pt',map_location=torch.device('cpu'))) #  new
        self.net.to(self.device)
        self.pub = rospy.Publisher("object_pose", Float64MultiArray, queue_size=1)
        self.run_flag = True
        self.model = torch.load('/maskrcnn/trained/model_50_2.pth',map_location=torch.device('cpu'))
        self.model.to(self.device)
        self.model.eval()
        print('****************************   start spin   ************************')
        self.run()
        rospy.spin()



    def run(self):
        while self.run_flag:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            # 读取图像
            depth = np.asanyarray(aligned_depth_frame.get_data())
            rgb = np.asanyarray(color_frame.get_data())
            # cv2.imwrite('/home/sunh/6D_ws/MPGrasp/work_space/data/depth.png',depth)
            # cv2.imwrite('/home/sunh/6D_ws/MPGrasp/work_space/data/rgb.png',rgb)

            img_o = rgb.copy()

            ################## fast rcnn ######################
            img = rgb.copy()
            img = torch.from_numpy(img.transpose((2, 0, 1)))
            img = img.float().div(255)
            with torch.no_grad():
                prediction = self.model([img.to(self.device)])
                boxes = prediction[0]['boxes']
                scores = prediction[0]['scores']

            all_point = []
            all_bbox = []
            for idx in range(boxes.shape[0]):
                if scores[idx] >= 0.5:
                    x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
                    x, y = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
                    # cv2.circle(img_o, (x, y), 3, (255, 0, 0), 2)
                    # cv2.circle(img_o, (x, y), 40, (0, 255, 0), 2)
                    all_point.append((x, y))
                    all_bbox.append((x1, y1, x2, y2))
                    # cv2.rectangle(img_show, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=2)

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
                    if dis < 120:
                        dis_m += dis
                        score += 1

                all_dis.append(dis_m)
                all_score.append(score)

            # all_score = np.array(all_score)
            # index = np.argmin(all_score)
            # print(all_score)
            # print(all_dis)
            # all_dis = np.array(all_dis)
            index = np.argmin(all_dis)
            # cv2.circle(img_o, (all_point[index][0], all_point[index][1]), 3, (0, 0, 255), 2)  # 抓取关键点1
            # index = 0
            x1, y1, x2, y2 = all_bbox[index]
            rgb = rgb[int(y1):int(y2), int(x1): int(x2)]
            w, h = int(x2) - int(x1), int(y2) - int(y1),
            # print(x1, y1, x2, y2)
            ################## fast rcnn ######################

            # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # depth = (depth * 0.00012498664727900177 ).astype(np.float32)
            depth = (depth / 1000.0).astype(np.float32)


            with torch.no_grad():
                #### input image
                img = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_NEAREST)

                img_fusion = gen_multi_feature_fusion_map(img)
                img_x = F.to_pil_image(img.astype(np.uint8))
                img_x = preprocess_rgb(img_x)
                img_fusion = F.to_pil_image(img_fusion.astype(np.uint8))
                img_fusion = preprocess_rgb(img_fusion)

                img_x = img_x.unsqueeze(0)
                # img_x = img_x.type(torch.cuda.FloatTensor)
                img_x = img_x.type(torch.FloatTensor)
                img_fusion = img_fusion.unsqueeze(0)
                # img_fusion = img_fusion.type(torch.cuda.FloatTensor)
                img_fusion = img_fusion.type(torch.FloatTensor)
                img_input = torch.cat([img_x, img_fusion], dim=1)

                #### model forward
                self.net.forward(img_input)

                #### get output
                pre_pose, pre_path = self.net.read_network_output()
                pre_pose = gaussian(pre_pose, 2.0, preserve_range=True)

                local_max_pose = peak_local_max(pre_pose, min_distance=20, threshold_abs=0.2,
                                                num_peaks=2)  # 在抓取关键点图中找到两个局部最大点
                g_point = []
                for grasp_point_array in local_max_pose:
                    grasp_point = tuple(grasp_point_array)
                    g_point.append(grasp_point)

                if len(g_point)<2 :
                    continue

                grasp_kps_map = np.zeros((128, 128, 3))
                cv2.circle(grasp_kps_map, (g_point[0][1], g_point[0][0]), 3, (255, 0, 0), 2)  # 抓取关键点1
                cv2.circle(grasp_kps_map, (g_point[1][1], g_point[1][0]), 3, (0, 255, 0), 2)  # 抓取关键点2
                # cv2.circle(grasp_kps_map, (local_max_path[0][1], local_max_path[0][0]), 2, (0, 0, 255), 2)  # 抓取关键点center
                cv2.circle(grasp_kps_map, (int((g_point[0][1] + g_point[1][1])/2), int((g_point[0][0] + g_point[1][0])/2)), 2, (0, 0, 255), 2)  # 抓取关键点center

                ## get grasp score
                # score1, score2 = util.eval_grasp(local_max_path, g_point)
                # print(score1,score2)
                # if score1 >0.8 or score2>0.8:
                #     continue

                ## get grasp key point and center
                grasp_kps_1, grasp_kps_2, grasp_center = util.process_grasp_kps_center(w, h, grasp_kps_map)

                ## 计算抓取关键点连线与水平线夹角，作为抓取角度
                dx = grasp_kps_2[0] - grasp_kps_1[0]
                dy = grasp_kps_2[1] - grasp_kps_1[1]
                angle = (np.arctan2(-dy, dx) + np.pi / 2) % np.pi - np.pi / 2


                depth_x = int((x1+x2)/2.0) #grasp_center[0]+x1
                depth_y =  int((y1+y2)/2.0) #grasp_center[1]+y1
                # depth_x = int((grasp_kps_1[0] + grasp_kps_2[0]) / 2.0 + x1)
                # depth_y = int((grasp_kps_1[1] + grasp_kps_2[1]) / 2.0 + y1)
                cv2.circle(img_o, (grasp_kps_1[0]+x1, grasp_kps_1[1]+y1), 3, (255, 0, 0), 2)
                cv2.circle(img_o, (grasp_kps_2[0]+x1, grasp_kps_2[1]+y1), 3, (0, 255, 0), 2)
                cv2.circle(img_o, (depth_x, depth_y), 3, (0, 0, 255), 2)

                if angle > 0:
                    angle_grasp = angle - np.pi/2
                if angle<0:
                    angle_grasp = angle + np.pi/2

                detect_result = Float64MultiArray()
                point_depth = depth[depth_y, depth_x]

                pose = uv2xyz((depth_x, depth_y), point_depth)
                detect_result.data = [pose[0],pose[1],pose[2] ,angle_grasp, 1]
                # print('pose: ', pose)
                # print('angle_grasp: ', angle_grasp)
                cv2.imshow('pre_p', pre_pose)
                # cv2.imshow('pre_path', pre_path)
                cv2.imshow('img_o', img_o)
                cv2.imshow('rgb', rgb)
                # cv2.waitKey(1)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
                self.pub.publish(detect_result)


if __name__ == '__main__':
    rospy.init_node("detect_realtime_node")
    while not rospy.is_shutdown():
        try:
            detect_node = DetectNode()
            rospy.spin()
            # break
        except KeyboardInterrupt:
            break

