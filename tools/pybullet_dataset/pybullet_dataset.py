import cv2
import pybullet as p
from time import sleep
import numpy as np
import math
import numpy as np
import pybullet_data
from PIL import Image
from scipy.spatial.transform import Rotation as R
import random
from bop_toolkit_lib import renderer



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
    planeId = p.loadURDF("/home/sunh/miniconda3/envs/grasp/lib/python3.7/site-packages/pybullet_data/plane.urdf")
    trayUid = p.loadURDF(
        "./tools/pybullet_dataset/mesh/tray/sunhan2.urdf",
        basePosition=[0, 0, 0])


###############################################   pyrender 配置  ############################################
random_light = False # 随机灯光
render_near = 0.1   # 渲染最近距离，单位是毫米
render_far = 10000 # 渲染最与远距离，单位是毫米
K = np.array(  [[621.399658203125, 0, 313.72052001953125],
              [0, 621.3997802734375, 239.97579956054688],
              [0, 0, 1]]) ## 相机内参
fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
IM_W, IM_H = 640, 480  # 渲染的图像大小
# 渲染零件的ply模型（可通过solidworks获得）
CAD_MODE_PATH = './work_space/mesh/1/obj_1.ply'

# 渲染器
obj_id = 0
renderer_type = "vispy"
ambient_weight = 0.1  # Weight of ambient light [0, 1]
shading = "phong"  # 'flat', 'phong'
ren_rgb = renderer.create_renderer(IM_W, IM_H, renderer_type, mode="rgb", shading=shading)
ren_rgb.set_light_ambient_weight(ambient_weight)
ren_rgb.add_object(obj_id, CAD_MODE_PATH)

(width_depth,height_depth,) = ( IM_W, IM_H,)
ren_depth = renderer.create_renderer(width_depth, height_depth, renderer_type, mode="depth")
ren_depth.add_object(obj_id, CAD_MODE_PATH)

###############################################   pybullet 配置  ############################################
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)  ## 设置重力
## 读取物料框和地板的urdf文件到pybullet
planeId = p.loadURDF("/home/sunh/miniconda3/envs/grasp/lib/python3.7/site-packages/pybullet_data/plane.urdf")
trayUid=p.loadURDF("./tools/pybullet_dataset/mesh/tray/sunhan2.urdf",basePosition=[0,0,0])
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
obj_number = 20


for idx in range(5):
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
                omron = p.loadURDF("./work_space/mesh/1/obj_01/urdf/obj_01.urdf",
                                   [ret_x, ret_y, 0.002 * idx + 0.005 * i + 0.1])  # idx * i * 0.05 + 0.2
            else:
                omron = p.loadURDF("./work_space/mesh/1/obj_01/urdf/obj_01.urdf",
                                   [0, 0, 0.0004 * idx + 0.006 * i + 0.1])  # idx * i * 0.05 + 0.2
            boxId_all.append(omron)

        # 仿真步数目，此处2000，该数值越大，仿真时间越久，2000的时候，零件落下并且位姿稳定
        for i in range(2000):
            p.stepSimulation()

        for i in range(obj_number):
            ## 读取每个零件在世界坐标下的位姿，并转换到相机坐标系
            cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId_all[i])
            T_obj_camera = pybullet_pose2gl(cubeOrn, cubePos, T_camera_world)
            T_obj_camera2 = pybullet_pose2gl(cubeOrn, cubePos, T_camera_world2)
            # 开始渲染，（渲染两次是因为有两个相机位置）
            R_render =  T_obj_camera[:3,:3]
            t_render =  T_obj_camera[:3,3]

            bgr1 = ren_rgb.render_object(
                obj_id, R_render, t_render, fx, fy, cx, cy)["rgb"]
            depth1 = ren_depth.render_object(
                obj_id, R_render, t_render, fx, fy, cx, cy)["depth"]


            R_render =  T_obj_camera2[:3,:3]
            t_render =  T_obj_camera2[:3,3]
            bgr2 = ren_rgb.render_object(
                obj_id, R_render, t_render, fx, fy, cx, cy)["rgb"]
            depth2 = ren_depth.render_object(
                obj_id, R_render, t_render, fx, fy, cx, cy)["depth"]
            
            ## 计算2D包围框，并去除无法出现在渲染视野中的位姿态
            ys1, xs1 = np.nonzero(depth1 > 0)
            try:
                obj_bb1 = calc_2d_bbox(xs1, ys1, (640, 480))
            except ValueError as e:
                print('Object in Rendering not visible. Have you scaled the vertices to mm? 111111111')
            if len(ys1) ==  0  and len(xs1) == 0:
                print('Obj not visible 1')
            else:
                if  0 < obj_bb1[1] < 405:
                    RT.append(T_obj_camera)

            print('obj_bb1: ',obj_bb1)
            mask = (depth1 > 1e-8).astype('uint8')
            show_msk = (mask / mask.max() * 255).astype("uint8")
            cv2.imshow('s',bgr2)
            cv2.waitKey(1)
            ## 计算2D包围框，并去除无法出现在渲染视野中的位姿态
            ys2, xs2 = np.nonzero(depth2 > 0)
            try:
                obj_bb2 = calc_2d_bbox(xs2, ys2, (640, 480))
            except ValueError as e:
                print('Object in Rendering not visible. Have you scaled the vertices to mm? 22222222222')
            if len(ys2) ==  0  and len(xs2) == 0:
                print('Obj not visible 2')
            else:
                if 0 < obj_bb2[1] < 405:
                    RT.append(T_obj_camera2)

        print('>>>>>>>>>>>>>>>>>: {}'.format(idx))

#最终保存
np.save('./work_space/mesh/0/RT.npy', RT)
print('saved')
