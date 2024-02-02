"""
Create the RT which can full of the whole space []
Author: sunhan
"""
import numpy as np
import cv2
from lib.pysixd_stuff import view_sampler
import math
from lib.meshrenderer import meshrenderer_phong
import random
# ##############################################   config：start  ############################################
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
ply_model_paths = [str('./work_space/mesh/8/obj_8.ply')]
max_rel_offset = 0.2  # used change the abs bbox
# ##############################################   config：end  ############################################
#
Renderer = meshrenderer_phong.Renderer(ply_model_paths,samples=1,vertex_scale=float(1)) # float(1) for some models

# sunhan add : get the bbox from depth
def calc_2d_bbox(xs, ys, im_size):
    bbTL = (max(xs.min() - 1, 0),
            max(ys.min() - 1, 0))
    bbBR = (min(xs.max() + 1, im_size[0] - 1),
            min(ys.max() + 1, im_size[1] - 1))
    return [bbTL[0], bbTL[1], bbBR[0] - bbTL[0], bbBR[1] - bbTL[1]]

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

# R1 = view_sampler.sample_views(600, radius=600,azimuth_range=(0,  2 * math.pi),elev_range=( 0, math.pi)    )
# R2 = view_sampler.sample_views(600, radius=600,azimuth_range=(0,  2 * math.pi),elev_range=( -math.pi, 0)    )
# ############################################  for vis my r   ########################################
# RT = []
# # t_all = np.array([[0,0,800],[0,0,700] , [0,0,600] ])
# t_all = np.array([0,0,400])
# # for i in range(len(t_all)):
# for i in range(1):
#     for idx in range(len(R1[0]) + len(R2[0])):
#         if idx < len(R1[0]):
#             R = R1[0][idx]['R']
#             # t = t_all[i]
#             t = t_all -np.array([0,0,50])
#             print('R1 ------------------')
#         else:
#             R = R2[0][idx - len(R1[0])]['R']
#             # t = t_all[i]
#             t = t_all -np.array([0,0,50])
#             print('R1 +++++++++++++++++++++')
#
#         bgr, depth = Renderer.render(
#             obj_id=0,
#             W=IM_W,
#             H=IM_H,
#             K=K.copy(),
#             R=R,
#             t=t,
#             near=render_near,
#             far=render_far,
#             random_light=random_light
#         )
#         mask = (depth > 1e-8).astype('uint8')
#         show_msk = (mask / mask.max() * 255).astype("uint8")
#         cv2.imshow('bgr', bgr)
#         cv2.waitKey(1)
#
#         T = np.identity(4)
#         T[:3, : 3] = R
#         T[0:3, 3] = t.reshape(3)
#         RT.append(T)
#         print('R all is saved!:  {} '.format(idx))
# np.save('/home/sunh/6D_ws/AAE_torch/tools/RT_4panel.npy', RT)



###############################  sampling for panelpose   ###############################
RT=[]
t_all = np.array([[0,0,400] , [0,0,420],[0,0,380],[0,0,440] ])
for i_t in range(len(t_all)):
    for idx in range(100):
        theta_x = random.uniform(-math.pi / 8, math.pi / 8)
        theta_y = random.uniform(-math.pi / 8, math.pi / 8)
        theta_z = random.uniform(-math.pi, math.pi)
        x = random.uniform(-15, 15)
        y = random.uniform(-15, 15)

        if idx < 50:
            theta = np.array([0, 0, (idx/36)*math.pi])
        else:
            # theta = np.array([0, 0, (idx/36)*math.pi])
            theta = np.array([theta_x, theta_y, theta_z])

        R = angle2Rmat(theta)



        t = t_all[i_t] + np.array( [x,y,180] )

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

        mask = (depth > 1e-8).astype('uint8')
        show_msk = (mask / mask.max() * 255).astype("uint8")
        cv2.imshow('bgr', bgr)
        cv2.waitKey(0)

        T = np.identity(4)
        T[:3, : 3] = R
        T[0:3, 3] = t.reshape(3)
        RT.append(T)
        print('R all is saved!:  {} '.format(idx))


np.save('./work_space/mesh/8/RT_4grasp.npy', RT)






















