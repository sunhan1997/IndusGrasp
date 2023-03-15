"""
This dataset.py generate Synthetic Data including train_image and heatmaps.
"""
import glob
import random
from lib.pysixd_stuff import view_sampler
from lib.aeutils import lazy_property
from lib.meshrenderer import meshrenderer, meshrenderer_phong
import configparser
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils.kps_heatmap as kps_heatmap


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




class Dataset(object):

    def __init__(self, dataset_path):
        H = 128 # resize后图像的高
        W = 128 # resize后图像的宽
        C = 3   # resize后图像的通道数

        self.numb_bg_imgs = 50    # 背景图像个数
        self.train_imgs_per_obj = int(400)  # 每个物体渲染图像个数

        self.shape = (int(H), int(W), int(C)) # 图像shape
        self.dataset_path = dataset_path  # 数据放置路径
        self.bg_img_paths = glob.glob('/media/sunh/Samsung_T5/6D_data/other_code/PVNET/SUN/JPEGImages/*.jpg') # 读取背景图像（用于随即背景）
        self.render_dims = (480, 640) # 渲染图像维度

        # self.obj_id = [1,2,3,4,5,6,7]
        self.obj_id = [0,1,]  # 训练物体的ID
        self.imgs_numb_all = self.train_imgs_per_obj * int(len(self.obj_id))  # 所有物体图像总数

        self.train_x = np.empty( (self.imgs_numb_all,) + self.shape, dtype=np.uint8 ) # 保存训练图像
        self.depth = np.empty( (self.imgs_numb_all,) + self.shape[:2], dtype=np.float32 ) # 保存训练图像 depth
        self.mask = np.empty( (self.imgs_numb_all,)  + self.shape[:2], dtype= bool) # 保存训练图像mask
        self.pose = np.empty( (self.imgs_numb_all,) + self.shape[:2], dtype=np.float32 ) # 保存训练的输出-抓取关键点
        self.gs_path_with = np.empty( (self.imgs_numb_all,) + self.shape[:2], dtype=np.float32) # 保存训练的输出-抓取路径
        self.bg_imgs = np.empty( (self.numb_bg_imgs,) + self.shape, dtype=np.uint8 )  # 保存背景图像


        self.K = [621.399658203125, 0, 313.72052001953125, 0, 621.3997802734375, 239.97579956054688, 0, 0, 1]    ## 相机内参，影响不大

        cfg_file_path = './work_space/render.cfg'  ## 域随即化配置文件
        args = configparser.ConfigParser()
        args.read(cfg_file_path)
        dataset_args = {k: v for k, v in
                        args.items('Dataset') +
                        args.items('Paths') +
                        args.items('Augmentation') +
                        args.items('Queue') +
                        args.items('Embedding') +
                        args.items('Training')}
        self.code = dataset_args['code']

        self.obj_bb = []
        self.keypoints = []

    ## 创建训练所需图像
    def get_training_images(self, dataset_path):
        current_file_name = os.path.join(dataset_path, 'train_data' + '.npz')
        if os.path.exists(current_file_name):  # 从已有的数据集中读取训练数据
            training_data = np.load(current_file_name)
            self.train_x = training_data['train_x'].astype(np.uint8)
            self.depth = training_data['depth']
            self.mask = training_data['mask']
            self.pose = training_data['pose']
            self.gs_path_with = training_data['gs_path_with']
        else: ## 创建训练数据
            intrinsic_matrix = np.array(self.K).reshape(3, 3) # 相机内参

            for idx in range(len(self.obj_id)):
                obj_id = self.obj_id[idx]
                ply_model_paths = [str('./work_space/mesh/{}/obj_{}.ply'.format(obj_id,obj_id))] # 读取物体PLY文件
                grasp_file = "./work_space/mesh/{}/grasp_path_point.txt".format(obj_id) # 读取物体抓取路径txt
                center_file = "./work_space/mesh/{}/obj_color_center.txt".format(obj_id) # 读取物体抓取关键点
                if obj_id == 8 :
                    self.render_RT_all = np.load('./work_space/mesh/8/RT_4grasp.npy')  ## 载入渲染时需要的位姿
                # if obj_id == 7 :
                #     self.render_RT_all = np.load('./mesh/7/RT_4grasp.npy')  ## 载入渲染时需要的位姿
                else:
                    self.render_RT_all = np.load('./work_space/mesh/RT_4grasp.npy')
                self.render_training_images_input(idx,ply_model_paths,grasp_file, center_file,intrinsic_matrix)  # 渲染训练图像的输入输出


            np.savez(current_file_name, train_x=self.train_x, mask=self.mask, pose=self.pose, gs_path_with =  self.gs_path_with, depth = self.depth )#保存训练数据
        print('loaded %s training images' % len(self.train_x))

    ## 载入背景图像
    def load_bg_images_sunhan(self, dataset_path):
            current_file_name = os.path.join(dataset_path, 'bg_image.npy')
            print('---------------------: ', current_file_name)
            if os.path.exists(current_file_name):
                self.bg_imgs = np.load(current_file_name)
            else:
                file_list = self.bg_img_paths[:self.numb_bg_imgs]
                for j, fname in enumerate(file_list):
                    fname = file_list[j]
                    print('loading bg img %s/%s' % (j, self.numb_bg_imgs))
                    bgr = cv2.imread(fname)
                    # bgr = bgr[int(166):int(166+50), int(283): int(283+50)]
                    bgr = cv2.resize(bgr, (self.shape[1],self.shape[0]))
                    cv2.imshow('bgr', bgr)
                    cv2.waitKey()
                    self.bg_imgs[j] = bgr
                np.save(current_file_name, self.bg_imgs)
            print('loaded %s bg images' % self.numb_bg_imgs)

    
    def render_training_images_input(self,obj_idx,ply_model_paths, grasp_file, center_file, intrinsic_matrix):
        H, W = self.shape[0],self.shape[1]
        K = np.array(self.K).reshape(3, 3)
        clip_near = 0.1
        clip_far = 10000

        render_x = meshrenderer_phong.Renderer( ply_model_paths, 1, self.dataset_path, 1)

        for idx in range( 0 + obj_idx * self.train_imgs_per_obj, self.train_imgs_per_obj + obj_idx * self.train_imgs_per_obj ):
            RT = self.render_RT_all[idx -  obj_idx * self.train_imgs_per_obj]
            R = RT[:3, :3]
            t = RT[0:3, 3] #- np.array((0,0,200))


            bgr, depth = render_x.render(
                obj_id=0,
                W=self.render_dims[1],
                H=self.render_dims[0],
                K=K.copy(),
                R=R,
                t=t,
                near=clip_near,
                far=clip_far,
                random_light=True
            )
            rgb_img = bgr.copy()


            ###### 找到BBOX ######
            ys, xs = np.nonzero(depth > 0)
            try:
                obj_bb = view_sampler.calc_2d_bbox(xs, ys, self.render_dims)
            except ValueError as e:
                print('Object in Rendering not visible. Have you scaled the vertices to mm?')
                break



            ## obj_bbx 添加 offset
            obj_bb_off = obj_bb  +  np.array([-4,-4, 8, 8])
            mask = (depth > 1e-8).astype('uint8')

            ## 图像剪裁和resize
            x_start, y_start, w, h = obj_bb_off
            mask = mask[int(y_start):int(y_start + h), int(x_start): int(x_start + w)]
            depth = depth[int(y_start):int(y_start + h), int(x_start): int(x_start + w)]
            bgr = bgr[int(y_start):int(y_start + h), int(x_start): int(x_start + w)]
            bgr = cv2.resize(bgr, (128, 128), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (128, 128), interpolation=cv2.INTER_NEAREST)

            show_msk = (mask / mask.max() * 255).astype("float32")
            cv2.imshow('bgr',bgr)
            cv2.imshow('mask_x',show_msk)
            cv2.imshow('depth',depth)
            cv2.waitKey(1)
            
            # 保存图像
            self.train_x[idx] = bgr.astype(np.uint8)
            self.depth[idx] = depth
            self.mask[idx] = mask
            self.obj_bb.append(obj_bb_off)
            print('input image: %s/%s' % (idx, self.train_imgs_per_obj + obj_idx * self.train_imgs_per_obj))

            ##################################################### 渲染输出图像 ################################################################3
            print('create_GT_masks: {}'.format(idx))

            pose = np.zeros((480, 640))

            RT = RT[0:3, 0:4]

            # 读取抓取路径点数据并根据RT投影到二维图像
            pt_grasp = np.loadtxt(grasp_file, usecols=(0, 1, 2))
            for i in range(len(pt_grasp)):
                kpt_hep = []
                pt_cld_data_point = pt_grasp[[i]]
                ones = np.ones((pt_cld_data_point.shape[0], 1))
                homogenous_coordinate = np.append(pt_cld_data_point[:, :3], ones, axis=1)

                # Perspective Projection to obtain 2D coordinates for masks
                homogenous_2D = intrinsic_matrix @ (
                        RT @ homogenous_coordinate.T)
                coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
                coord_2D = ((np.floor(coord_2D)).T).astype(int)
                x_2d = np.clip(coord_2D[:, 0], 0, 639)
                y_2d = np.clip(coord_2D[:, 1], 0, 479)
                # pose[y_2d, x_2d] = 1
                land = np.array((x_2d[0],y_2d[0]))
                kpt_hep.append(land)
                heatmap = kps_heatmap.generate_heatmaps(kpt_hep, 480, 640)
                pose[heatmap[0]>0] = 1 

            # 读取抓取路径 中心 点数据并根据RT投影到二维图像
            pt_center = np.loadtxt(center_file, usecols=(0, 1, 2))
            center_heatmap = np.zeros((480, 640,len(pt_center)))
            keypoint = []
            for i in range(len(pt_center)):
                kpt_hep = []
                pt_cld_data_point = pt_center[[i]]
                ones = np.ones((pt_cld_data_point.shape[0], 1))
                homogenous_coordinate = np.append(pt_cld_data_point[:, :3], ones, axis=1)

                # Perspective Projection to obtain 2D coordinates for masks
                homogenous_2D = intrinsic_matrix @ (
                        RT @ homogenous_coordinate.T)
                coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
                coord_2D = ((np.floor(coord_2D)).T).astype(int)
                x_2d = np.clip(coord_2D[:, 0], 0, 639)
                y_2d = np.clip(coord_2D[:, 1], 0, 479)
                rgb_img[y_2d, x_2d] = 255
                land = np.array((x_2d[0],y_2d[0]))
                kpt_hep.append(land)
                keypoint.append(land)
                heatmap = kps_heatmap.generate_heatmaps(kpt_hep, 480, 640)  #(17,17)
                center_heatmap[:,:,i] = heatmap[0]

            ## 剪裁和resize
            pose = pose[int(y_start):int(y_start + h), int(x_start): int(x_start + w)]
            center_heatmap = center_heatmap[int(y_start):int(y_start + h), int(x_start): int(x_start + w)]
            pose = cv2.resize(pose, (128, 128), interpolation=cv2.INTER_NEAREST)
            center_heatmap = cv2.resize(center_heatmap, (128, 128), interpolation=cv2.INTER_NEAREST)

            ## 抓取关键点mask，其中抓取关键点k赋值为5，物体mask赋值为0.5
            pose_mask = show_msk.copy()
            pose_mask[pose_mask>0] = 0.5
            pose_mask[pose>0] = 1

            for i in range(len(pt_center)):
                pose_mask[center_heatmap[:,:,i]>0] = 5
            # pose_mask[center_heatmap[:,:,1]>0] = 6

            ## 抓取路径mask，其中抓取路径赋值为5，物体mask赋值为0.5
            path_mask = show_msk.copy()
            path_mask[path_mask>0] = 0.5
            path_mask[pose>0] = 5


            cv2.imshow('pose_mask', pose_mask)
            cv2.imshow('rgb_img', rgb_img)
            cv2.waitKey(1)
            self.pose[idx] = pose_mask
            self.gs_path_with[idx] = path_mask



    @lazy_property
    def _aug(self):
        from imgaug.augmenters import Sequential,SomeOf,OneOf,Sometimes,WithColorspace,WithChannels, \
            Noop,Lambda,AssertLambda,AssertShape,Scale,CropAndPad, \
            Pad,Crop,Fliplr,Flipud,Superpixels,ChangeColorspace, PerspectiveTransform, \
            Grayscale,GaussianBlur,AverageBlur,MedianBlur,Convolve, \
            Sharpen,Emboss,EdgeDetect,DirectedEdgeDetect,Add,AddElementwise, \
            AdditiveGaussianNoise,Multiply,MultiplyElementwise,Dropout, \
            CoarseDropout,Invert,ContrastNormalization,Affine,PiecewiseAffine, \
            ElasticTransformation
        return eval(self.code)

    @lazy_property
    def _aug_occl(self):
        from imgaug.augmenters import Sequential,SomeOf,OneOf,Sometimes,WithColorspace,WithChannels, \
            Noop,Lambda,AssertLambda,AssertShape,Scale,CropAndPad, \
            Pad,Crop,Fliplr,Flipud,Superpixels,ChangeColorspace, PerspectiveTransform, \
            Grayscale,GaussianBlur,AverageBlur,MedianBlur,Convolve, \
            Sharpen,Emboss,EdgeDetect,DirectedEdgeDetect,Add,AddElementwise, \
            AdditiveGaussianNoise,Multiply,MultiplyElementwise,Dropout, \
            CoarseDropout,Invert,ContrastNormalization,Affine,PiecewiseAffine, \
            ElasticTransformation
        return Sequential([Sometimes(0.7, CoarseDropout( p=0.4, size_percent=0.01) )])


    ## RGB图像获得多特征融合图，作为输入
    def gen_multi_feature_fusion_map(self, img, test_model=False):
        img_fld = fld(img.copy(), test_model)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_edge = cv2.Canny(img, 50, 200)

        img_gray = np.expand_dims(img_gray, axis=2)
        img_fld = np.expand_dims(img_fld, axis=2)
        img_edge = np.expand_dims(img_edge, axis=2)

        img_add = np.concatenate([img_gray, img_edge], 2)
        img_add = np.concatenate([img_add, img_fld], 2)

        return img_add


    ## 把数据放入网络
    def torch_batch(self, index):
        ## load 背景数据
        assert self.numb_bg_imgs > 0
        rand_idcs_bg = np.random.choice(self.numb_bg_imgs, 1, replace=False)
        ## load 训练输入数据
        batch_x, masks, depths = self.train_x[index], self.mask[index], self.depth[index]/1000.0
        ## load 训练输出数据
        pose = self.pose[index]
        path = self.gs_path_with[index]

        ## add background
        # rand_idcs_bg = rand_idcs_bg.tolist()
        # rand_vocs = self.bg_imgs[rand_idcs_bg]
        # rand_vocs[masks] = batch_x[masks]
        # batch_x = rand_vocs
        # batch_x = self._aug.augment_images(batch_x)

        batch_x = batch_x[0]
        # 训练数据获得多特征融合图
        batch_fusion = self.gen_multi_feature_fusion_map(batch_x)

        # cv2.imwrite('/home/sunh/6D_ws/MPGrasp/work_space/data/{}.png'.format(index),batch_x)
        # img_edge = cv2.Canny(batch_x, 50, 200)
        # cv2.imwrite('/home/sunh/6D_ws/MPGrasp/work_space/data/{}.jpg'.format(index),img_edge)

        # ## debug
        # cv2.imshow('batch_x', batch_x)
        # cv2.waitKey()
        # cv2.imshow('batch_fusion', batch_fusion)
        # cv2.waitKey()
        # plt.imshow(pose[0])
        # # plt.imshow(pose)
        # # plt.title('pose')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(path[0])
        # # plt.imshow(path)
        # # plt.title('pose')
        # plt.colorbar()
        # plt.show()

        
        return (batch_x,batch_fusion, pose, path )



# dataset_path = './work_space/data'
# dataset = Dataset(dataset_path)
# dataset.load_bg_images_sunhan(dataset_path)   # get bg image
# dataset.get_training_images(dataset_path)  # get train image
# dataset.torch_batch([1])  # get train image





















