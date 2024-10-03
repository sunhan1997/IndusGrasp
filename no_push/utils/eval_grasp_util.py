import math

import numpy as np
import cv2
import matplotlib.pyplot as plt


def getDist_P2L_sunhan( point_center,  pointA,  pointB):
    A = (pointA[1] - pointB[1]) / (pointA[0] - pointB[0])
    B = pointA[1] - A * pointA[0]
    distance = ( np.abs(A*point_center[0] + B - point_center[1] )) / ( np.sqrt(A*A + B*B) )
    return distance


def get_Slope(  pointA,  pointB):
    A = (pointA[1] - pointB[1]) / (pointA[0] - pointB[0])
    return np.abs(A)

def eval_grasp(point_center, g_point):
    score_dis = getDist_P2L_sunhan((point_center[0][1], point_center[0][0]), (g_point[1][1], g_point[1][0]),
                                   (g_point[0][1], g_point[0][0]))
    score_angle1 = get_Slope((point_center[0][1], point_center[0][0]), (g_point[0][1], g_point[0][0]))
    score_angle2 = get_Slope((point_center[0][1], point_center[0][0]), (g_point[1][1], g_point[1][0]))
    score_angle_align = np.abs(score_angle1 - score_angle2)
    # print('score_dis:  ', score_dis)
    # print('score_angle_align:  ', score_angle_align)

    return score_dis, score_angle_align


def process_grasp_kps_center(w,h,grasp_kps_map):
    grasp_kps_map = cv2.resize(grasp_kps_map, (w, h), interpolation=cv2.INTER_NEAREST)  # 图像变回原来尺寸

    point1 = np.where(grasp_kps_map[:, :, 0] == 255)
    point2 = np.where(grasp_kps_map[:, :, 1] == 255)
    center = np.where(grasp_kps_map[:, :, 2] == 255)
    gy1 = point1[0][int(len(point1[0]) / 2)]
    gx1 = point1[1][int(len(point1[0]) / 2)]
    gy2 = point2[0][int(len(point2[0]) / 2)]
    gx2 = point2[1][int(len(point2[0]) / 2)]
    gy = center[0][int(len(center[0]) / 2)]
    gx = center[1][int(len(center[0]) / 2)]

    grasp_kps_1 = np.array([gx1,gy1])
    grasp_kps_2 = np.array([gx2,gy2])
    grasp_center = np.array([gx,gy])

    return grasp_kps_1, grasp_kps_2, grasp_center

def process_grasp_kps_center_circle(w,h,grasp_kps_map,rgb2):
    grasp_kps_map = cv2.resize(grasp_kps_map, (w, h), interpolation=cv2.INTER_NEAREST)  # 图像变回原来尺寸

    point1 = np.where(grasp_kps_map[:, :, 0] == 255)
    point2 = np.where(grasp_kps_map[:, :, 1] == 255)
    cneter = np.where(grasp_kps_map[:, :, 2] == 255)
    gy1 = point1[0][int(len(point1[0]) / 2)]
    gx1 = point1[1][int(len(point1[0]) / 2)]
    gy2 = point2[0][int(len(point2[0]) / 2)]
    gx2 = point2[1][int(len(point2[0]) / 2)]
    gy = cneter[0][int(len(cneter[0]) / 2)]
    gx = cneter[1][int(len(cneter[0]) / 2)]

    grasp_kps_1 = np.array([gx1,gy1])
    grasp_kps_2 = np.array([gx2,gy2])
    grasp_center = np.array([gx,gy])

    return grasp_kps_1, grasp_kps_2, grasp_center


def get_grasp_points(grasp_kps_1, grasp_kps_2 ):
    # pre_pose = cv2.resize(pre_pose, (w, h), interpolation=cv2.INTER_NEAREST)  # 图像变回原来尺寸
    # cv2.line(pre_pose, (grasp_kps_1[0],grasp_kps_1[1]), (grasp_kps_2[0],grasp_kps_2[1]),(0, 0, 255), 1)
    # cv2.imshow('pre_pose_grasp_line',pre_pose)
    # cv2.waitKey()
    path_Slope = get_Slope(grasp_kps_1, grasp_kps_2)
    b = grasp_kps_1[1] - grasp_kps_1[0] * path_Slope
    grasp_points = []
    for i in range( np.abs(grasp_kps_2[0] - grasp_kps_1[0]) ):
        if grasp_kps_1[0] < grasp_kps_2[0]:
            point = np.array((grasp_kps_1[0] + i, (grasp_kps_1[0] + i) * path_Slope + b))
        else:
            point = np.array((grasp_kps_1[0] - i, (grasp_kps_1[0] - i) * path_Slope + b))
        grasp_points.append(point)
    return grasp_points


def get_grasp_with(pre_pose, w,h,grasp_point, grasp_kps_1, grasp_kps_2):
    pre_pose = cv2.resize(pre_pose, (w, h), interpolation=cv2.INTER_NEAREST)  # 图像变回原来尺寸

    path_Slope = get_Slope((grasp_kps_1[0] , grasp_kps_1[1] ), (grasp_kps_2[0] , grasp_kps_2[1]))
    b = (grasp_point[0] ) + (grasp_point[1] ) / path_Slope
    path_Slope = -1/path_Slope
    point1 = np.array((grasp_point[0] - 10, (grasp_point[0] - 10) * path_Slope + b))
    point2 = np.array((grasp_point[0] + 10, (grasp_point[0] + 10) * path_Slope + b))
    grasp_mask = np.zeros((h,w,3))
    cv2.line(grasp_mask, (int(point1[0]),int(point1[1])), (int(point2[0]),int(point2[1])),(1, 1, 1), 2)
    pre_pose = pre_pose * grasp_mask[:,:,0]

    grasp_coor = np.where(pre_pose>0)
    griper1 = np.array((grasp_coor[1][0],grasp_coor[0][0])) # x y
    griper2 = np.array((grasp_coor[1][len(grasp_coor[0])-1],grasp_coor[0][len(grasp_coor[0])-1]))  # x y

    return griper1,griper2

def uv2xyz(uv,z,cx = 316.4925842285156,cy = 231.94068908691406,fx=622.8344116210938,fy=622.83447265625):
    xcoord = (uv[0] - cx) * z / fx
    ycoord = (uv[1]- cy) * z/ fy
    return xcoord,ycoord,z

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

def fast_collision_detection(grasp_center, griper1, griper2, depth,x1,y1):
    point_depth = depth[int(grasp_center[1] + y1), int(grasp_center[0000] + x1)] * 0.00012498664727900177

    griper1 = depth[griper1[1], griper1[0] ] * 0.00012498664727900177
    griper2 = depth[griper2[1], griper2[0] ] * 0.00012498664727900177

    if griper1 > point_depth and  griper2 > point_depth:
        return True
    else:
        return False