import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode
from data import cfg_mnet
from utils.nms.py_cpu_nms import py_cpu_nms
import numpy as np
from torchvision.ops import RoIPool

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )



class Angle(nn.Module):
    def __init__(self,roi_size=40, spatial_scale=1):
        super(Angle, self).__init__()
        self.roi = RoIPool((roi_size, roi_size), spatial_scale)
        self.conv1 = conv_bn(3, 32, 1 )
        self.conv2 = conv_bn(32, 32, 2 )
        self.conv3 = conv_bn(32, 32, 1)
        self.conv4 = conv_bn(32, 32, 2)
        self.conv5 = conv_bn(32, 32, 1)
        self.conv6 = conv_bn(32, 32, 2)
        self.conv7 = conv_bn(32, 32, 1)
        self.conv8 = conv_bn(32, 32, 2)
        self.max_pool = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()  # 已经nms后筛选出来的ROI

        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] * x.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] * x.size()[2]
        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)

        roi_pool = self.roi(x, indices_and_rois)  # 300x1024x14x14
        out1 = self.conv1(roi_pool)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)
        out7 = self.conv7(out6)
        out8 = self.conv8(out7)
        out9 = self.max_pool(out8)
        out10 = out9.view(out9.size(0), -1)
        out11 = self.fc1(out10)

        rpy = self.fc2(out11)


        return rpy



class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)  # 2 classfiy


class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)  # 4 loc


class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*50,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 50)  # 50 landmark


def decode_face_box_out(out):
    batch_size_n = out[0].shape[0]
    loc = out[0]
    conf =out[1]
    priorbox = PriorBox(cfg_mnet, image_size=(cfg_mnet['image_size'], cfg_mnet['image_size']))  # 实例化一个先验框对象
    priors = priorbox.forward()  # 对象里面的方法（获取先验框坐标以及数量信息）
    priors = priors.to(device)
    prior_data = priors.data
    rois = list()
    roi_indices = list()
    for i in range(batch_size_n):
        box = decode(loc[i].data.cuda(), prior_data, torch.tensor(cfg_mnet['variance']).cuda())
        box = box.cpu().numpy()
        score = conf[i].data.cpu().numpy()[:, 1]  # 取出cof中序号为1的内容

        # keep top-K before NMS
        order = score.argsort()[::-1][:50]  # 对得分框从大到小排序(取前50个)
        box = box[order[0]]  # 取出分数最大的那个框
        score = score[order[0]] # 取出分数最大的那个分数
        tensor_roi = torch.from_numpy(box).unsqueeze(0)
        batch_index = i * torch.ones((1))  # 这里的1 是每个batchsize 只有一个人脸ROI
        roi_indices.append(batch_index)
        rois.append(tensor_roi)
    rois = torch.cat(rois, dim=0)
    roi_indices = torch.cat(roi_indices, dim=0)

    return rois, roi_indices

class Retinaface_Angle_Net(nn.Module):
    def __init__(self,cfg = None,train=0):
        super(Retinaface_Angle_Net, self).__init__()
        self.cfg = cfg
        retinaface_net = RetinaFace_Net(self.cfg).to(device)
        if train == 1:
            state_dict = torch.load("./weights/V4_320_130w/V4_320_epoch_34_best.pth")
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                head = k[:7]
                if head == 'module.':
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            retinaface_net.load_state_dict(new_state_dict)

        self.retinaface_net_backbone =retinaface_net
        self.angle = Angle()
    def forward(self, inputs):
        face_out,backbone_out = self.retinaface_net_backbone(inputs)
        backbone_out = list(backbone_out.values())
        backbone = inputs
        # backbone = backbone_out[0]

        img_size = backbone.shape[2:]

        rois, roi_indices = decode_face_box_out(face_out)
        angle_rpy = self.angle.forward(backbone, rois, roi_indices,img_size)

        return angle_rpy, face_out, rois

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class RetinaFace_Net(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace_Net,self).__init__()
        self.phase = phase
        backbone_mobilenetv1 = MobileNetV1().to(device)
        self.body = _utils.IntermediateLayerGetter(backbone_mobilenetv1, {'stage1': 1, 'stage2': 2, 'stage3': 3})
        in_channels_stage2 = 32
        in_channels_list = [         # 主干网络中三个不同大小的特征层：16 、16 、16，链接检测分支
            in_channels_stage2 * 1,  # 16
            in_channels_stage2 * 1,  # 16
            in_channels_stage2 * 1,  # 16
        ]
        out_channels = 32
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels = out_channels)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels = out_channels)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=out_channels)

    def _make_class_head(self,fpn_num=3,inchannels=32,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=32,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=32,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        backbone_out = self.body(inputs)

        # FPN
        fpn = self.fpn(backbone_out)

        # SSH
        feature1 = self.ssh1(fpn[0])  # 78x128x2
        feature2 = self.ssh2(fpn[1])  # 39x64x2
        feature3 = self.ssh3(fpn[2])  # 20x32x2
        features = [feature1, feature2, feature3]
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions,  F.softmax(classifications, dim=-1), ldm_regressions)
            #output = (bbox_regressions, classifications)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
            #output = (bbox_regressions, F.softmax(classifications, dim=-1))
        return output,backbone_out




class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])  # 80x80x64 -> 80x80x64
        output2 = self.output2(input[1])  # 40x40x128 -> 40x40x64
        output3 = self.output3(input[2])  # 20x20x256 -> 20x20x64
        # 20x20x64 -> 40x40x64
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)
        # 40x40x64 -> 80x80x64
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 16, 2, leaky = 0.1),    # 3
            conv_dw(16, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 32, 2),  # 27
            conv_dw(32, 32, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(32, 32, 2),  # 43 + 16 = 59
            conv_dw(32, 32, 1), # 59 + 32 = 91
            conv_dw(32, 32, 1), # 91 + 32 = 123
            conv_dw(32, 32, 1), # 123 + 32 = 155
            conv_dw(32, 32, 1), # 155 + 32 = 187
            conv_dw(32, 32, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(32, 32, 2), # 219 +3 2 = 241
            conv_dw(32, 32, 1), # 241 + 64 = 301
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

