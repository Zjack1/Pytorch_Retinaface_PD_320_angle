from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from nets.retinaface_net_V5 import RetinaFace
from utils.box_utils import decode, decode_landm_25,draw_landmark_line

import time

parser = argparse.ArgumentParser(description = 'Retinaface')
parser.add_argument('--trained_model', default='./weights/V5/V5_epoch_20.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=500, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=1, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()


def get_all_files(dir):
    files_ = []
    video_name =[]
    list = os.listdir(dir)
    for i in range(0, len(list)):
        video_name.append(list[i])
        path = os.path.join(dir, list[i])
        if os.path.isdir(path):
            files_.extend(get_all_files(path))
        if os.path.isfile(path):
            files_.append(path)
    return files_,video_name


img_path = r"\\10.1.1.125\Development\MAPA2\wissen_video\DMS_APP测试误报数据\阴阳脸误检"
img_path = r"C:\Users\shzhoujun\Desktop\Retinaface_PFPLD\data\test1"
img_path = r"C:\Users\shzhoujun\Desktop\Retinaface_PFPLD\data\test_shake_3w"
#img_path = r"\\10.1.1.125\Development\ShareFolder\DMS\Recruitment_center\test_pic"
all_test_image_path, all_images_name = get_all_files(img_path)
len_all_image_path_files = len(all_test_image_path)


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    # else:
    #     pretrained_dict = remove_prefix(pretrained_dict, 'module.')  # 删除module前缀
    # check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1

    # testing begin
    for i in range(len_all_image_path_files):
        print("image name = ", all_test_image_path[i])
        image_path = all_test_image_path[i]
        if all_images_name[i] == "Thumbs.db":
            continue
        img_raw = cv2.imdecode(np.fromfile(all_test_image_path[i], dtype=np.uint8),-1)
        img_resize = cv2.resize(img_raw, (160, 160))
        img = np.float32(img_resize)

        im_height, im_width, _ = img_resize.shape
        # 它的作用是将归一化后的框坐标转换成原图的大小
        scale = torch.Tensor([img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0]])

        img -= (104, 117, 123)
        #img /= 255.0

        img = img.transpose(2, 0, 1)  # 通道转换
        img = torch.from_numpy(img).unsqueeze(0)  # 增加batch size维度
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landmark = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))  # 实例化一个先验框对象
        priors = priorbox.forward()  # 对象里面的方法（获取先验框坐标以及数量信息）
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        landms = decode_landm_25(landmark.data.squeeze(0), prior_data, cfg['variance'])
        landms = landms.cpu().numpy()

        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]  # 取出cof中序号为1的内容
        all_cof = conf.squeeze(0).data.cpu().numpy()  # 取出cof中所有的内容

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        loc_boxes = loc.squeeze(0).data.cpu().numpy()[inds]
        scores = scores[inds]
        landms = landms[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]  # 对得分框从大到小排序
        boxes = boxes[order]
        scores = scores[order]
        landms = landms[order]

        # do NMS np.hstack()是把矩阵进行行连接（1773,4）+ （1773,1）= （1773,5）
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        last_landmark = landms[:args.keep_top_k, :]
        last_landmark_x = last_landmark[0][::2]*img_raw.shape[1]
        last_landmark_y = last_landmark[0][1::2]*img_raw.shape[0]

        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                print(b[4])
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                for landmk in range(25):
                    cv2.circle(img_raw, (last_landmark_x[landmk], last_landmark_y[landmk]), 1, (0, 0, 255), 4)
                img_raw = draw_landmark_line(img_raw,last_landmark_x,last_landmark_y)

            # save image
            cv2.imshow("test",img_raw)
            if cv2.waitKey(111) == 27:
                break
            name = "test.jpg"
            cv2.imwrite(name, img_raw)

