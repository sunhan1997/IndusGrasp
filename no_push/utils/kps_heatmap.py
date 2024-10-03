import os
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import PIL
from torchvision import transforms
import torch
from PIL import Image, ImageFont, ImageDraw


# json变成加入高斯的np
def json_to_numpy(dataset_path):
    with open(dataset_path) as fp:
        json_data = json.load(fp)
        points = json_data['shapes']

    # print(points)
    landmarks = []
    for point in points:
        for p in point['points']:
            landmarks.append(p)

    # print(landmarks)
    landmarks = np.array(landmarks)
    landmarks = landmarks.reshape(-1, 2)

    # 保存为np
    # np.save(os.path.join(save_path, name.split('.')[0] + '.npy'), landmarks)

    return landmarks


def generate_heatmaps(landmarks, height, width, sigma = (7,7)):

    heatmaps = []
    for points in landmarks:
        heatmap = np.zeros((height, width))
        heatmap[points[1]][points[0]] = 1
        heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
        am = np.amax(heatmap)
        heatmap /= am / 255
        heatmaps.append(heatmap)
    heatmaps = np.array(heatmaps)
    return heatmaps


def show_heatmap(heatmaps):
    for heatmap in heatmaps:
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.show()


def heatmap_to_point(heatmaps):
    points = []
    for heatmap in heatmaps:
        pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        points.append([pos[1], pos[0] ])
    return np.array(points)


def show_inputImg_and_keypointLabel(imgPath, heatmaps):
    points = []
    for heatmap in heatmaps:
        pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        points.append([pos[1], pos[0]])

    img = PIL.Image.open(imgPath).convert('RGB')
    img = transforms.ToTensor()(img)  # 3*3000*4096
    img = img[:, :cfg['cut_h'], :cfg['cut_w']]
    img = img.unsqueeze(0)  # 增加一维
    resize = torch.nn.Upsample(scale_factor=(0.25, 0.25), mode='bilinear', align_corners=True)
    img = resize(img)
    img = img.squeeze(0)  # 减少一维
    print(img.shape)
    img = transforms.ToPILImage()(img)
    draw = ImageDraw.Draw(img)
    for point in points:
        print(point)
        draw.point((point[0], point[1]), fill='yellow')
    # 保存
    img.save(os.path.join('..','show', 'out.jpg'))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
preprocess_rgb = transforms.Compose([
    # transforms.ColorJitter(brightness=32. / 255., contrast=0.5, saturation=0.5, hue=0.2), #TODO was 0.05
    transforms.ToTensor(),
    normalize
])


