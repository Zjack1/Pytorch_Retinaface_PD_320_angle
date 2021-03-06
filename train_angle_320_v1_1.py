from __future__ import print_function
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import detection_collate, preproc_angle, cfg_mnet, cfg_re50
from data.wider_face_angle import WiderFaceDetection
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from nets.retinaface_rpn_net_V4x320_conv_v1_1 import Retinaface_Angle_Net
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--train_img_path', default='../data2/ws_shake_3w/', help='training image path')
parser.add_argument('--training_dataset', default='../data2/train_ws_shake_3w_25_landmark_pry_list.txt', help='Training dataset directory')
# parser.add_argument('--train_img_path', default='../face_data_27w_retinaface_rpy/img_shake_27w/', help='training image path')
# parser.add_argument('--training_dataset', default='../face_data_27w_retinaface_rpy/train_shake_27w_25_landmark_pry_list.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default="./weights/27w_conv_v1/Angle_V4_conv_v1_27w_epoch_6.pth", help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/27w_conv_v1_1/', help='Location to save checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder
train_img_path = args.train_img_path

net = Retinaface_Angle_Net(cfg=cfg,train =1)
print("Printing net...")
print(net)



class SmoothL1(nn.Module):
    def __init__(self):
        super(SmoothL1, self).__init__()

    def forward(self,angle_gt , angle):

        pose_p = angle.view(-1, 3)
        pose_t = angle_gt.view(-1, 3)
        pose_loss = F.smooth_l1_loss(pose_p, pose_t, reduction='mean')
        return pose_loss * 10

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self,angle_gt, angle):
        pose_loss = torch.sum((angle_gt - angle) * (angle_gt - angle), dim=1)
        return torch.mean(pose_loss) * 10



#
# if args.resume_net is not None:
#     print('Loading resume network...')
#     state_dict = torch.load(args.resume_net)
#     # create new OrderedDict that does not contain `module.`
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         head = k[:7]
#         if head == 'module.':
#             name = k[7:] # remove `module.`
#         else:
#             name = k
#         new_state_dict[name] = v
#     net.load_state_dict(new_state_dict)
#


if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True

#optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
optimizer = optim.Adam(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
criterion = SmoothL1()

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = WiderFaceDetection(training_dataset, train_img_path, preproc_angle(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    a = net.retinaface_net_backbone.parameters()
    for param in net.retinaface_net_backbone.parameters():
        param.requires_grad = False
    net.freeze_bn()




    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 2 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + 'Angle_V4x320_conv_v1_1_27w_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]
        gt_angle = torch.cat(targets, dim=0).reshape(-1, 3)

        # forward
        angle_rpy, face_out, _ = net(images)

        # backprop
        optimizer.zero_grad()
        loss_angle  = criterion(angle_rpy, gt_angle)
        loss = loss_angle
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || loss_angle: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_angle.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
