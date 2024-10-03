"""
This dataset.py generate Synthetic Data including train_image and heatmaps.
"""
import glob
from utils.aeutils import lazy_property
import configparser
import os
import cv2
import numpy as np
from torchvision import transforms
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa

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



class Dataset_Train(object):

    def __init__(self, dataset_path):
        H = 128 # resize后图像的高
        W = 128 # resize后图像的宽
        C = 3   # resize后图像的通道数

        self.numb_bg_imgs = 100    # 背景图像个数
        self.train_imgs_per_obj = int(100)  # 每个物体渲染图像个数
        self.shape = (int(H), int(W), int(C)) # 图像shape

        # self.obj_id = [1,2,3,4,5,6,7]
        self.obj_id = [1,]  # 训练物体的ID
        self.imgs_numb_all = self.train_imgs_per_obj * int(len(self.obj_id))  # 所有物体图像总数

        self.train_x = np.empty( (self.imgs_numb_all,) + self.shape, dtype=np.uint8 ) # 保存训练图像
        self.mask = np.empty( (self.imgs_numb_all,)  + self.shape[:2], dtype= bool) # 保存训练图像mask
        self.pose = np.empty( (self.imgs_numb_all,) + self.shape[:2], dtype=np.float32 ) # 保存训练的输出-抓取关键点
        self.gs_path_with = np.empty( (self.imgs_numb_all,) + self.shape[:2], dtype=np.float32) # 保存训练的输出-抓取路径
        self.bg_imgs = np.empty( (self.numb_bg_imgs,) + self.shape, dtype=np.uint8 )  # 保存背景图像


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


    ## 创建训练所需图像
    def get_training_images(self, dataset_path):
        current_file_name = os.path.join(dataset_path, 'train_data' + '.npz')
        if os.path.exists(current_file_name):  # 从已有的数据集中读取训练数据
            training_data = np.load(current_file_name)
            self.train_x = training_data['train_x'].astype(np.uint8)
            self.mask = training_data['mask']
            self.pose = training_data['pose']
            self.gs_path_with = training_data['gs_path_with']
        else:
            print('lack train_data.npz !!!!!!!!!!!!!')

        print('loaded %s training images' % len(self.train_x))

    ## 载入背景图像
    def load_bg_images_sunhan(self, dataset_path):
            current_file_name = os.path.join(dataset_path, 'bg_image.npy')
            print('load bg_image:  ', current_file_name)
            self.bg_imgs = np.load(current_file_name)

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


    def add_random_light(self, image):
        brightness = np.random.randint(-100,100)
        image_with_light = cv2.add(image, brightness)
        return image_with_light

    ## 把数据放入网络
    def torch_batch(self, index):
        ## load 背景数据
        assert self.numb_bg_imgs > 0
        rand_idcs_bg = np.random.choice(self.numb_bg_imgs, 1, replace=False)
        batch_x, masks  = self.train_x[index], self.mask[index]
        pose = self.pose[index]
        path = self.gs_path_with[index]

        ## add background
        rand_idcs_bg = rand_idcs_bg.tolist()
        rand_vocs = self.bg_imgs[rand_idcs_bg]
        rand_vocs[masks] = batch_x[masks]
        # batch_x = rand_vocs
        # batch_x = self._aug.augment_images(batch_x)

        batch_x = batch_x[0]
        # batch_x = self.add_random_light(batch_x)
        # 训练数据获得多特征融合图
        batch_fusion = gen_multi_feature_fusion_map(batch_x)

        # cv2.imwrite('/media/sunh/HW/孙晗/2/{}.png'.format(index), batch_x)
        # # ## debug
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



class Dataset_Test(object):

    def __init__(self, dataset_path):
        H = 128  # resize后图像的高
        W = 128  # resize后图像的宽
        C = 3  # resize后图像的通道数

        self.numb_bg_imgs = 100  # 背景图像个数
        self.train_imgs_per_obj = int(100)  # 每个物体渲染图像个数
        self.shape = (int(H), int(W), int(C))  # 图像shape

        # self.obj_id = [1,2,3,4,5,6,7]
        self.obj_id = [1,]  # 训练物体的ID
        self.imgs_numb_all = self.train_imgs_per_obj * int(len(self.obj_id))  # 所有物体图像总数

        self.train_x = np.empty((self.imgs_numb_all,) + self.shape, dtype=np.uint8)  # 保存训练图像
        self.mask = np.empty( (self.imgs_numb_all,)  + self.shape[:2], dtype= bool) # 保存训练图像mask
        self.pose = np.empty((self.imgs_numb_all,) + self.shape[:2], dtype=np.float32)  # 保存训练的输出-抓取关键点
        self.gs_path_with = np.empty((self.imgs_numb_all,) + self.shape[:2], dtype=np.float32)  # 保存训练的输出-抓取路径
        self.bg_imgs = np.empty((self.numb_bg_imgs,) + self.shape, dtype=np.uint8)  # 保存背景图像

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


    def get_training_images(self, dataset_path):
        current_file_name = os.path.join(dataset_path, 'test_data' + '.npz')
        if os.path.exists(current_file_name):
            training_data = np.load(current_file_name)
            self.train_x = training_data['train_x'].astype(np.uint8)
            self.mask = training_data['mask']
            self.pose = training_data['pose']
            self.gs_path_with = training_data['gs_path_with']
        else:
            print('lack test_data.npz !!!!!!!!!!!!!')

    ## 载入背景图像
    def load_bg_images_sunhan(self, dataset_path):
            current_file_name = os.path.join(dataset_path, 'bg_image.npy')
            print('load bg_image: ', current_file_name)
            if os.path.exists(current_file_name):
                self.bg_imgs = np.load(current_file_name)
            else:
                print('lack bg_image.npz !!!!!!!!!!!!!')
            print('loaded %s bg images' % self.numb_bg_imgs)

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

    def torch_batch(self, index):
        ## load 背景数据
        assert self.numb_bg_imgs > 0
        rand_idcs_bg = np.random.choice(self.numb_bg_imgs, 1, replace=False)
        batch_x, masks  = self.train_x[index], self.mask[index]
        pose = self.pose[index]
        path = self.gs_path_with[index]

        ## add background
        rand_idcs_bg = rand_idcs_bg.tolist()
        rand_vocs = self.bg_imgs[rand_idcs_bg]
        rand_vocs[masks] = batch_x[masks]
        batch_x = rand_vocs
        batch_x = self._aug.augment_images(batch_x)

        batch_x = batch_x[0]
        # 训练数据获得多特征融合图
        batch_fusion = gen_multi_feature_fusion_map(batch_x)

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



class Train_DatasetLodar(Dataset_Train):
    def __init__(self,dataset_path):
        self.dataset = Dataset_Train(dataset_path)
        self.dataset.load_bg_images_sunhan(dataset_path)  # get bg image
        self.dataset.get_training_images(dataset_path)  # get train image
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.color_transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize])

    def __getitem__(self,index):
        img, img_fusion ,pose, path = self.dataset.torch_batch([index])
        img = F.to_pil_image(img.astype(np.uint8))
        img = self.color_transform (img)
        img_fusion = F.to_pil_image(img_fusion.astype(np.uint8))
        img_fusion = self.color_transform (img_fusion)

        pose = torch.tensor(pose, dtype=torch.float32)
        path = torch.tensor(path, dtype=torch.float32)
        return img,img_fusion, pose, path

    def __len__(self):
        return self.dataset.imgs_numb_all



class Test_DatasetLodar(Dataset_Test):
    def __init__(self,dataset_path):
        self.dataset = Dataset_Test(dataset_path)
        self.dataset.load_bg_images_sunhan(dataset_path)  # get bg image
        self.dataset.get_training_images(dataset_path)  # get train image
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.color_transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize])

    def __getitem__(self,index):
        img, img_fusion ,pose, path = self.dataset.torch_batch([index])
        img = F.to_pil_image(img.astype(np.uint8))
        img = self.color_transform (img)
        img_fusion = F.to_pil_image(img_fusion.astype(np.uint8))
        img_fusion = self.color_transform (img_fusion)

        pose = torch.tensor(pose, dtype=torch.float32)
        path = torch.tensor(path, dtype=torch.float32)
        return img,img_fusion, pose, path


    def __len__(self):
        return self.dataset.imgs_numb_all

