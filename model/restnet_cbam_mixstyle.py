import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model.unet_parts import up, outconv, outconv_depth
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from model.mixstyle import MixStyle2 as MixStyle




__all__ = ['ResNet', 'resnet18']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}

# 注意力机制 CBAM
class Channel_Attention(nn.Module):   # CAM

    def __init__(self, channel, r=16):
        super(Channel_Attention, self).__init__()
        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._max_pool = nn.AdaptiveMaxPool2d(1)

        self._fc = nn.Sequential(
            nn.Conv2d(channel, channel // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // r, channel, 1, bias=False)
        )

        self._sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self._avg_pool(x)   # avg pooling
        y1 = self._fc(y1)

        y2 = self._max_pool(x)   # max pooling
        y2 = self._fc(y2)

        y = self._sigmoid(y1 + y2)  # add sigmoid
        return x * y                # scale


# 注意力机制 CBAM
class Spatial_Attention(nn.Module):

    def __init__(self, kernel_size=3):
        super(Spatial_Attention, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2

        self._layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)    # avg pool in every pixel
        max_mask, _ = torch.max(x, dim=1, keepdim=True)  # max pool in every pixel
        mask = torch.cat([avg_mask, max_mask], dim=1)    # concat

        mask = self._layer(mask)   # conv
        return x * mask            # scale

# 注意力机制 SE
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)   # sequeeze
        y = self.fc(y).view(b, c, 1, 1)   # expansion

        return x * y.expand_as(x)         # scale



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False, cbam=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        ##### add CBAM attention
        self.se = se
        self.cbam = cbam
        self.se_layer = SELayer(planes, 16)
        self.ca_layer = Channel_Attention(planes, 16)
        self.sa_layer = Spatial_Attention()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        ##### add CBAM attention
        if self.se and not self.cbam:  # se
            out = self.se_layer(out)
        if not self.se and self.cbam:  # cbam
            out = self.ca_layer(out)
            out = self.sa_layer(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False, cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        ##### add CBAM attention
        self.se = se
        self.cbam = cbam
        self.ca_layer = Channel_Attention(planes * self.expansion, 16)
        self.sa_layer = Spatial_Attention()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        ##### add CBAM attention
        if self.se and not self.cbam:  # se
            out = self.se_layer(out)
        if not self.se and self.cbam:  # cbam
            out = self.ca_layer(out)
            out = self.sa_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def _freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False





# 主要网络
class ResNet(nn.Module):

    def __init__(self, block, layers, mixstyle_layers=[], mixstyle_p=0.5, mixstyle_alpha=0.3 ):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 编码器 RESNET
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 解码器 输出 抓取关键点
        self.up1_p = up(384, 128)
        self.up2_p = up(192, 64)
        self.up3_p = up(128, 64)
        self.up4_p = up(64, 64, add_shortcut=False)

        self.up3_pa = up(128, 64)
        self.up4_pa = up(64, 64, add_shortcut=False)

        self.huber_loss = torch.nn.HuberLoss(reduction='mean')

        # mixstyle
        self.mixstyle = None
        if mixstyle_layers:
            self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
            for layer_name in mixstyle_layers:
                assert layer_name in ["conv1", "conv2_x", "conv3_x", "conv4_x", "conv5_x"]
            print("Insert MixStyle after the following layers: {}".format(mixstyle_layers))
        self.mixstyle_layers = mixstyle_layers


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 解码器最后一层
        self.out_pose = outconv(64, 1)
        self.out_path = outconv(64, 1)


        _freeze_module(self.conv1)
        _freeze_module(self.bn1)
        _freeze_module(self.layer1)

        # 优化器
        parameters_to_train = list(filter(lambda p: p.requires_grad, self.parameters()))
        self.optimizer = optim.Adam(parameters_to_train, weight_decay=0.00004, lr=0.00005)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.maxpool(x1)

        x3 = self.layer1(x2)
        if "conv2_x" in self.mixstyle_layers:
            x3 = self.mixstyle(x3)

        x3 = self.layer2(x3)
        if "conv3_x" in self.mixstyle_layers:
            x3 = self.mixstyle(x3)

        x4 = self.layer3(x3)
        # if "conv4_x" in self.mixstyle_layers:
        #     x4 = self.mixstyle(x4)

        # 解码器 = pose
        p = self.up1_p(x4, x3)
        p1 = self.up2_p(p, x2)

        p = self.up3_p(p1, x1)
        p = self.up4_p(p,x)
        # 解码器 = path
        pa = self.up3_pa(p1, x1)
        pa = self.up4_pa(pa,x)

        pose_map = self.out_pose(p)
        path_map = self.out_path(pa)

        self.pre_pose = pose_map ## 抓取关键点
        self.pre_path = path_map ## 抓取关键点



    def optimize(self, rgb, pose, path ):

        self.train()
        self.forward(rgb)
        self.zero_grad()


        self.loss_pose = self.huber_loss(self.pre_pose, pose)
        self.loss_path = self.huber_loss(self.pre_path, path)

        self.loss =   self.loss_pose   + self.loss_path *3
        self.loss.backward()
        self.optimizer.step()
        return self.loss_pose, self.loss_path

    def read_network_output(self):
        ## 输出
        pre_pose = self.pre_pose.detach().cpu().numpy().squeeze()
        pre_path = self.pre_path.detach().cpu().numpy().squeeze()
        return  pre_pose, pre_path






def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock,
                   [2, 2, 2, 2],
                   mixstyle_layers=["conv2_x", "conv3_x", "conv4_x"],
                   mixstyle_p=0.5,
                   mixstyle_alpha=0.1, **kwargs)  # resnet 18
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model



def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)  # resnet 34
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)  # resnet 50
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model

def resnet100(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)  # resnet 101
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = resnet18(False)
    x = torch.randn((2, 6, 128, 128))
    model.to(device)
    x = x.to(device)
    x = model(x)
    pass
