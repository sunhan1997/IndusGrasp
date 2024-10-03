import utils
import transforms as T
from engine import train_one_epoch, evaluate
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import os
import torch
import numpy as np
import torch.utils.data
from PIL import Image
from xml.dom.minidom import parse

def readXML_bbox(filepath, target):
    domTree = parse(filepath)
    rootNode = domTree.documentElement
    sizes = rootNode.getElementsByTagName("size")
    width = int(sizes[0].getElementsByTagName("width")[0].childNodes[0].data)
    height = int(sizes[0].getElementsByTagName("height")[0].childNodes[0].data)
    objects = rootNode.getElementsByTagName("object")
    output=[]
    for  i  in range(len(objects)):
               xmin=[]
               ymin=[]
               xmax=[]
               ymax=[]
               if objects[i].getElementsByTagName("name")[0].childNodes[0].data ==target:
                  xmin=float(objects[i].getElementsByTagName("xmin")[0].childNodes[0].data)
                  ymin=float(objects[i].getElementsByTagName("ymin")[0].childNodes[0].data)
                  xmax=float(objects[i].getElementsByTagName("xmax")[0].childNodes[0].data)
                  ymax=float(objects[i].getElementsByTagName("ymax")[0].childNodes[0].data)
                  output.append([xmin,ymin,xmax,ymax])
    return output


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "ori"))))
        self.anno = list(sorted(os.listdir(os.path.join(root, "ann"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "ori", self.imgs[idx])
        ann_path = os.path.join(self.root, "ann", self.anno[idx])
        img = Image.open(img_path).convert("RGB")

        outputs = readXML_bbox(ann_path, str('omron')) ##### should be changed
        boxes = []
        labels = []
        iscrowd = []
        for i in range(len(outputs)):
            boxes.append(outputs[i])
            labels.append(int(1))
            iscrowd.append(int(1))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# use the PennFudan dataset and defined transformations
dataset = MyDataset('/home/sunh/6D_grasp/IndusGrasp/maskrcnn/grasp1/', get_transform(train=True))
dataset_test = MyDataset('/home/sunh/6D_grasp/IndusGrasp/maskrcnn/grasp1/', get_transform(train=False))

torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:6])
dataset_test = torch.utils.data.Subset(dataset_test, indices[5:])

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2

model = get_instance_segmentation_model(num_classes)
# model = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/grasp6/model_50.pth')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

num_epochs = 200
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    # evaluate(model, data_loader_test, device=device)

    if (epoch + 1) % 10 == 0:
        model_name = "/home/sunh/6D_grasp/IndusGrasp/maskrcnn/trained/model_" + str(epoch + 1) + ".pth"
        torch.save(model, model_name)
        print("save model!!")
