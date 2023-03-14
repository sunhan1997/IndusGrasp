"""
Create the RT which can full of the whole space []
Author: sunhan
"""
import numpy as np
import cv2
import math
from lib.meshrenderer import meshrenderer_phong
from utils import ImgPcldUtils
import os

img_pcld_utils = ImgPcldUtils()

# ##############################################   OPENGL 配置  ############################################
FOR_R = True
VIS = True
bbox_VIS = False
random_light = True
render_near = 0.1
render_far = 10000 # unit: should be mm
K = np.array(  [[621.399658203125, 0, 313.72052001953125],
              [0,621.3997802734375, 239.97579956054688],
              [0, 0, 1]])
IM_W, IM_H = 640, 480
ply_model_paths = [str('/home/sunh/6D_ws/MPGrasp/work_space/help_code/0/obj_0.ply')]
max_rel_offset = 0.2  # used change the abs bbox
# ##############################################   OPENGL 配置  ############################################
#
Renderer = meshrenderer_phong.Renderer(ply_model_paths,samples=1,vertex_scale=float(1)) # float(1) for some models

# 计算bbox
def calc_2d_bbox(xs, ys, im_size):
    bbTL = (max(xs.min() - 1, 0),
            max(ys.min() - 1, 0))
    bbBR = (min(xs.max() + 1, im_size[0] - 1),
            min(ys.max() + 1, im_size[1] - 1))
    return [bbTL[0], bbTL[1], bbBR[0] - bbTL[0], bbBR[1] - bbTL[1]]

# 角度 to 旋转矩阵
def angle2Rmat(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


###############################  对物体的抓取路径和抓取关键点作采样   ###############################
###############################  注意：有的时候需要把物体转一个角度才能找到合适的位姿作采样   ###############################
RT=[]
t_all = np.array([[0,0,300] ])
for i_t in range(len(t_all)):
    for idx in range(1):

        theta = np.array([0, 0, 0])
        # theta = np.array([0, 90, 0])
        R = angle2Rmat(theta)
        t = t_all[i_t] #+ np.array( [x,y,0] )
  
        # 开始渲染
        bgr, depth = Renderer.render(
            obj_id=0,
            W=IM_W,
            H=IM_H,
            K=K.copy(),
            R=R,
            t=t,
            near=render_near,
            far=render_far,
            random_light=random_light
        )
        bgr = bgr/1.2
        mask = (depth > 1e-8).astype('uint8')
        show_msk = (mask / mask.max() * 255).astype("uint8")

        g_x = []
        g_y = []
        g_c = []

        mask_fordpt_p = np.zeros((depth.shape[0], depth.shape[1]))  ## 创建一个新的mask图，把抓取路径上的点放进去，方便后面乘以深度图
        mask_fordpt_c = np.zeros((depth.shape[0], depth.shape[1]))   ## 创建一个新的mask图，把抓取路径上中心点点放进去，方便后面乘以深度图
        def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                g_x.append(x)
                g_y.append(y)
                xy = "%d,%d" % (x, y)
                # cv2.circle(bgr, (x, y), 1, (255, 0, 0), thickness=-1)
                cv2.circle(bgr, (x, y), 1, (0, 255, 0), thickness=-1)
                cv2.putText(bgr, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                            1.0, (0, 0, 0), thickness=1)
                if len(g_x) % 2 == 0:
                    cv2.line(bgr, (g_x[len(g_x) - 2], g_y[len(g_x) - 2]), (g_x[len(g_x) - 1], g_y[len(g_x) - 1]),
                             (0, 0, 255), 1)
                    # center = np.array([( (g_x[len(g_x) - 2]+g_x[len(g_x) - 1])/2, (g_y[len(g_x) - 2]+g_y[len(g_x) - 1])/2  )])
                    # cv2.circle(bgr, (int(center[0][0]), int(center[0][1])), 3, (0, 255, 0), 2)
                    point = np.where(bgr[:, :, 2] == 255)
                    center_point = np.where(bgr[:, :, 1] == 255)
                    mask_fordpt_p[point[0], point[1]] = 1
                    mask_fordpt_c[center_point[0], center_point[1]] = 1
                cv2.imshow("image", bgr)

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)

        while (1):
            cv2.imshow("image", bgr)
            if cv2.waitKey(0) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
      
        # 通过深度图，获得抓取路径上点在模型上三维点的坐标
        depth_p = depth.copy()
        depth_p = depth_p * mask_fordpt_p
        depth_show = depth_p.copy()
        depth_show[depth_show > 0] = 255
        cv2.imshow('depth_p', depth_show)
        cv2.waitKey(0)
        kp_xyz = img_pcld_utils.dpt_2_cld(depth_p, 1, K)
        kp_xyz, msk = img_pcld_utils.filter_pcld(kp_xyz)
        kp_xyz = (kp_xyz - t).dot(R)

        # 通过深度图，获得抓取路径中心点在模型上三维点的坐标
        depth_c = depth.copy()
        depth_c = depth_c * mask_fordpt_c
        depth_show = depth_c.copy()
        depth_show[depth_show > 0] = 255
        cv2.imshow('depth_c', depth_show)
        cv2.waitKey(0)
        kp_xyz_c = img_pcld_utils.dpt_2_cld(depth_c, 1, K)
        kp_xyz_c, msk = img_pcld_utils.filter_pcld(kp_xyz_c)
        kp_xyz_c = (kp_xyz_c - t).dot(R)

     
        # 保存
        textured_fps_pth = os.path.join('/home/sunh/6D_ws/MPGrasp/work_space/help_code/0/grasp_path_point.txt')
        with open(textured_fps_pth, 'w') as of:
            for p3d in kp_xyz:
                print(p3d[0], p3d[1], p3d[2], file=of)

        textured_fps_pth = os.path.join('/home/sunh/6D_ws/MPGrasp/work_space/help_code/0/obj_color_center.txt')
        with open(textured_fps_pth, 'w') as of:
            for p3d in kp_xyz_c:
                print(p3d[0], p3d[1], p3d[2], file=of)



