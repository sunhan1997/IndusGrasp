import cv2
import pybullet as p
from time import sleep
import numpy as np
import math
import numpy as np
import pybullet_data
from PIL import Image
from scipy.spatial.transform import Rotation as R
from lib.meshrenderer import meshrenderer_phong
import random


## 计算图像中物体包围框
def calc_2d_bbox(xs, ys, im_size):
    bbTL = (max(xs.min() - 1, 0),
            max(ys.min() - 1, 0))
    bbBR = (min(xs.max() + 1, im_size[0] - 1),
            min(ys.max() + 1, im_size[1] - 1))
    return [bbTL[0], bbTL[1], bbBR[0] - bbTL[0], bbBR[1] - bbTL[1]]

## 从pybullet中得到零件的点云，暂时没有用到
def write_pointcloud(filename,pointCloud,img_h,img_w,stepX,stepY):
    # Write header of .ply file
    fid = open(filename,'w')
    fid.write(str('ply\n'))
    fid.write(str('format ascii 1.0\n'))
    fid.write(str('element vertex %d\n'%(pointCloud.size/4)))
    fid.write(str('property float x\n'))
    fid.write(str('property float y\n'))
    fid.write(str('property float z\n'))
    fid.write(str('end_header\n'))
    # Write 3D points to .ply file
    for h in range(0, img_h, stepY):
        for w in range(0, img_w, stepX):
                fid.write(str(pointCloud[h][w][0]) + " " + str(pointCloud[h][w][1])+ " " +
                 str(pointCloud[h][w][2]) + "\n")
    fid.close()

## 从pybullet中得到图像以及深度图，mask等等
def render(mode='human'):
    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0.],
                                                      distance=.5,
                                                      yaw=0,
                                                      pitch=-90,
                                                      roll=0, upAxisIndex=2)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(640) / 480,
                                               nearVal=0.1,
                                               farVal=100000.0)
    (_, _, rgb, depth, mask) = p.getCameraImage(width=640, height=480,
                                        viewMatrix=view_matrix,
                                        projectionMatrix=proj_matrix,
                                        renderer=p.ER_BULLET_HARDWARE_OPENGL)

    rgb_array = np.array(rgb, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (480, 640, 4))
    rgb_array = rgb_array[:, :, :3]

    projectionMatrix = np.asarray(proj_matrix).reshape([4, 4], order='F')
    viewMatrix = np.asarray(view_matrix).reshape([4, 4], order='F')
    tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))
    return rgb_array,depth,mask,tran_pix_world

## 从pybullet中得到图像以及深度图，mask等等
def pybullet_pose2gl(cubeOrn,cubePos,T_camera_world):
    Rm = R.from_quat([cubeOrn[0], cubeOrn[1], cubeOrn[2], cubeOrn[3]])
    rotation_matrix = Rm.as_matrix()
    T_obj_world = np.identity(4)
    T_obj_world[:3, :3] = rotation_matrix
    T_obj_world[0, 3] = cubePos[0] *1000
    T_obj_world[1, 3] = cubePos[1] *1000
    T_obj_world[2, 3] = cubePos[2] *1000
    T_world_camera = np.linalg.inv(T_camera_world)
    T_obj_camera = np.dot(T_world_camera, T_obj_world)
    return T_obj_camera

## 重新设置pybullet环境（每一轮零件散乱洒落后，都需要重新设置环境）
def reset():
    p.resetSimulation()
    # p.configureDebugVisualizer()
    p.setGravity(0, 0, -9.8)
    ## 读取物料框和地板的urdf文件到pybullet
    planeId = p.loadURDF("./mesh/plane.urdf",
                         basePosition=[0, 0, 0])
    trayUid = p.loadURDF("./mesh/tray/sunhan.urdf",
                         basePosition=[0, 0, 0])


###############################################   OPENGL 配置  ############################################
random_light = False # 随机灯光
render_near = 0.1   # 渲染最近距离，单位是毫米
render_far = 10000 # 渲染最与远距离，单位是毫米
K = np.array(  [[621.399658203125, 0, 313.72052001953125],
              [0, 621.3997802734375, 239.97579956054688],
              [0, 0, 1]]) ## 相机内参
IM_W, IM_H = 640, 480  # 渲染的图像大小
# 渲染零件的ply模型（可通过solidworks获得）
ply_model_paths = [str('/home/sunh/github/IndusGrasp/work_space/mesh/0/obj_0.ply')]
# 渲染器
Renderer = meshrenderer_phong.Renderer(ply_model_paths,samples=1,vertex_scale=float(1)) # float(1) for some models


###############################################   pybullet 配置  ############################################
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)  ## 设置重力
## 读取物料框和地板的urdf文件到pybullet
planeId = p.loadURDF("./mesh/plane.urdf",basePosition=[0, 0, 0])
trayUid=p.loadURDF("./mesh/tray/sunhan.urdf", basePosition=[0, 0, 0])
useRealTimeSimulation = 0

####  get T_camera_world （设置相机在pybullet环境世界的位置）
theta = (-180 * 3.1415926) / 180.
transformation = np.array(
    [[math.cos(theta), 0, math.sin(theta), 0], [0, 1, 0, 0], [-math.sin(theta), 0, math.cos(theta), 500], [0, 0, 0, 1]])

theta1 = (-180 * 3.1415926) / 180.
transformation2 = np.array(
    [[math.cos(theta1), 0, math.sin(theta1), 0], [0, 1, 0, 0], [-math.sin(theta1), 0, math.cos(theta1), 600], [0, 0, 0, 1]])

theta2 = (180 * 3.1415926) / 180
R_z = np.array([[math.cos(theta2), -math.sin(theta2), 0, 0],
                [math.sin(theta2), math.cos(theta2), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
                ])
T_camera_world = np.dot(R_z, transformation)
T_camera_world2 = np.dot(R_z, transformation2)


RT = []
boxId_all = []
obj_number = 35


for idx in range(100):
    if (useRealTimeSimulation):
        p.setGravity(0, 0, -9.8)
        sleep(0.01)  # Time in seconds.
    else:
        reset()
        rgb, depth, mask, tran_pix_world = render() # 得到pybullet中环境，暂未实用
        # 一次载入十个零件
        for i in range(obj_number):
            ret_x = random.uniform(-0.05, 0.05)
            ret_y = random.uniform(-0.05, 0.05)
            if idx < 300:
                # omron = p.loadURDF("/home/sunh/6D_ws/MPGrasp/tools/pybullet_dataset/mesh/omron2/urdf/omron1.SLDPRT.urdf",
                omron = p.loadURDF("./mesh/宝塔接头 3分14.SLDPRT/urdf/宝塔接头 3分14.SLDPRT.STL.urdf",
                                   [ret_x, ret_y, 0.002 * idx + 0.003 * i + 0.04])  # idx * i * 0.05 + 0.2
            else:
                omron = p.loadURDF("./mesh/宝塔接头 3分14.SLDPRT/urdf/宝塔接头 3分14.SLDPRT.STL.urdf",
                                   [0, 0, 0.0004 * idx + 0.006 * i + 0.1])  # idx * i * 0.05 + 0.2
            boxId_all.append(omron)

        # 仿真步数目，此处2000，该数值越大，仿真时间越久，2000的时候，零件落下并且位姿稳定
        for i in range(2000):
            # p.configureDebugVisualizer()
            p.stepSimulation()

        # for i in range(obj_number):
        #     ## 读取每个零件在世界坐标下的位姿，并转换到相机坐标系
        #     cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId_all[i])
        #     T_obj_camera = pybullet_pose2gl(cubeOrn, cubePos, T_camera_world)
        #     T_obj_camera2 = pybullet_pose2gl(cubeOrn, cubePos, T_camera_world2)
        #     # 开始渲染，（渲染两次是因为有两个相机位置）
        #     R_render =  T_obj_camera[:3,:3]
        #     t_render =  T_obj_camera[:3,3]
        #     bgr1, depth1 = Renderer.render(
        #         obj_id=0,
        #         W=IM_W,
        #         H=IM_H,
        #         K=K.copy(),
        #         R=R_render,
        #         t=t_render,
        #         near=render_near,
        #         far=render_far,
        #         random_light=random_light
        #     )
        #
        #     R_render =  T_obj_camera2[:3,:3]
        #     t_render =  T_obj_camera2[:3,3]
        #     bgr2, depth2 = Renderer.render(
        #         obj_id=0,
        #         W=IM_W,
        #         H=IM_H,
        #         K=K.copy(),
        #         R=R_render,
        #         t=t_render,
        #         near=render_near,
        #         far=render_far,
        #         random_light=random_light
        #     )
        #
        #     ## 计算2D包围框，并去除无法出现在渲染视野中的位姿态
        #     ys1, xs1 = np.nonzero(depth1 > 0)
        #     try:
        #         obj_bb1 = calc_2d_bbox(xs1, ys1, (640, 480))
        #     except ValueError as e:
        #         print('Object in Rendering not visible. Have you scaled the vertices to mm? 111111111')
        #     if len(ys1) ==  0  and len(xs1) == 0:
        #         print('Obj not visible 1')
        #     else:
        #         if  0 < obj_bb1[1] < 405:
        #             RT.append(T_obj_camera)
        #
        #     print('obj_bb1: ',obj_bb1)
        #     mask = (depth1 > 1e-8).astype('uint8')
        #     show_msk = (mask / mask.max() * 255).astype("uint8")
        #     cv2.imshow('s',bgr2)
        #     cv2.waitKey(1)
        #     ## 计算2D包围框，并去除无法出现在渲染视野中的位姿态
        #     ys2, xs2 = np.nonzero(depth2 > 0)
        #     try:
        #         obj_bb2 = calc_2d_bbox(xs2, ys2, (640, 480))
        #     except ValueError as e:
        #         print('Object in Rendering not visible. Have you scaled the vertices to mm? 22222222222')
        #     if len(ys2) ==  0  and len(xs2) == 0:
        #         print('Obj not visible 2')
        #     else:
        #         if 0 < obj_bb2[1] < 405:
        #             RT.append(T_obj_camera2)

        print('>>>>>>>>>>>>>>>>>: {}'.format(idx))

#最终保存
# np.save('./RT.npy', RT)
# print('saved')


####  get RGB / DEPTH / MASK from pybullet
# rgb,depth,mask, tran_pix_world = render()
# cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
# mask[mask <= 1] = 0
# mask[mask > 0] = 255
# cv2.imwrite('/home/sunh/6D_ws/AAE_torch/lib/pybullet_dataset/rgb.png',rgb)
# cv2.imwrite('/home/sunh/6D_ws/AAE_torch/lib/pybullet_dataset/depth.png',depth)
# cv2.imwrite('/home/sunh/6D_ws/AAE_torch/lib/pybullet_dataset/mask.png',mask)

####  get the pointcloud from sence
# imgH =480
# imgW =640
# for h in range(0, imgH):
#     for w in range(0, imgW):
#         x = (2 * w - imgW) / imgW
#         y = -(2 * h - imgH) / imgH  # be careful！ deepth and its corresponding position
#         z = 2 * depth[h, w] - 1
#         pixPos = np.asarray([x, y, z, 1])
#         print('pixPos: ', pixPos)
#
#         position = np.matmul(tran_pix_world, pixPos)
#         print('position: ', position)
#         pointCloud[np.int(h / 1), np.int(w / 1), :] = position / position[3]
# write_pointcloud("fromSideAngle3.ply", pointCloud, imgH, imgW, 1, 1)


##  start render
# R_render =  T_obj_camera[:3,:3]
# t_render =  T_obj_camera[:3,3]
# bgr, depth = Renderer.render(
#     obj_id=0,
#     W=IM_W,
#     H=IM_H,
#     K=K.copy(),
#     R=R_render,
#     t=t_render,
#     near=render_near,
#     far=render_far,
#     random_light=random_light
# )
# cv2.imshow('bgr',bgr)
# cv2.imwrite('/home/sunh/6D_ws/AAE_torch/lib/pybullet_dataset/bgr.png',bgr)
# cv2.waitKey(1)


# cubePos1, cubeOrn1 = p.getBasePositionAndOrientation(boxId_1)
# cubePos2, cubeOrn2 = p.getBasePositionAndOrientation(boxId_2)
# cubePos3, cubeOrn3 = p.getBasePositionAndOrientation(boxId_3)
# cubePos4, cubeOrn4 = p.getBasePositionAndOrientation(boxId_4)
# T_obj_camera0 = pybullet_pose2gl(cubeOrn0, cubePos0, T_camera_world)
# T_obj_camera1 = pybullet_pose2gl(cubeOrn1, cubePos1, T_camera_world)
# T_obj_camera2 = pybullet_pose2gl(cubeOrn2, cubePos2, T_camera_world)
# T_obj_camera3 = pybullet_pose2gl(cubeOrn3, cubePos3, T_camera_world)
# T_obj_camera4 = pybullet_pose2gl(cubeOrn4, cubePos4, T_camera_world)
# RT.append(T_obj_camera0)
# RT.append(T_obj_camera1)
# RT.append(T_obj_camera2)
# RT.append(T_obj_camera3)
# RT.append(T_obj_camera4)
