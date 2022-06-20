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
from nets.retinaface_rpn_net_V4x320_conv_v1_1 import Retinaface_Angle_Net
from utils.box_utils import decode, decode_landm_25,draw_landmark_line

import time


def get_all_files(dir):
    files_ = []
    list = os.listdir(dir)
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isdir(path):
            files_.extend(get_all_files(path))
        if os.path.isfile(path):
            files_.append(path)
    return files_, list

weights_path = "./weights/130w_conv_v1_1"
root = os.path.dirname(os.path.realpath(__file__))
all_weights_file, all_weights_name = get_all_files(weights_path)
len_all_weights_file = len(all_weights_file)




for i in range(len_all_weights_file):
    trained_model = all_weights_file[i]
    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument('--trained_model', default=trained_model,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.5, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=50, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.5, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=1, type=int, help='keep_top_k')
    parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    args = parser.parse_args()
    txt_path = r"\\10.1.1.125\Development\ShareFolder\DMS\Recruitment_center\Recruitment_center_test_landmark_rpy_list.txt"
    f = open(txt_path, 'r')
    lines = f.readlines()


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


    def bb_intersection_over_union(boxA, boxB):
        boxA = [int(x) for x in boxA]
        boxB = [int(x) for x in boxB]

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou


    if __name__ == '__main__':
        torch.set_grad_enabled(False)
        cfg = None
        if args.network == "mobile0.25":
            cfg = cfg_mnet
        elif args.network == "resnet50":
            cfg = cfg_re50
        # net and model
        net = Retinaface_Angle_Net(cfg=cfg)
        net = load_model(net, args.trained_model, args.cpu)
        net.eval()
        #print('Finished loading model!')
        #print(net)
        cudnn.benchmark = True
        device = torch.device("cpu" if args.cpu else "cuda")
        net = net.to(device)

        ious = 0
        landmark_losses = 0
        pry_losses = 0
        p_losses = 0
        r_losses = 0
        y_losses = 0
        len_file = len(lines)

        n = 0
        for line in lines:
            n = n + 1
            #print("--------", n)
            line = line.split(' ')
            img_path = line[0]
            img_raw = cv2.imread(img_path)
            img_resize = cv2.resize(img_raw, (320, 320))
            img = np.float32(img_resize)

            im_height, im_width, _ = img_resize.shape
            scale = torch.Tensor([img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0]])

            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            tic = time.time()
            angle_rpy, face_out, _ = net(img)  # forward pass
            loc, conf, landmark = face_out[0], face_out[1], face_out[2]
            #print('net forward time: {:.4f}'.format(time.time() - tic))

            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
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
            landmark_xy = last_landmark.reshape(-1, 2)
            landmark_xy_raw_size = landmark_xy * np.asarray([img_raw.shape[1], img_raw.shape[0]])
            last_landmark_x = last_landmark[0][::2] * img_raw.shape[1]
            last_landmark_y = last_landmark[0][1::2] * img_raw.shape[0]

            gt_box = line[1:5]
            iou = bb_intersection_over_union(dets[0], gt_box)
            ious = iou + ious

            gt_landmark = line[5:55]
            GT_landmark = torch.tensor([int(i) for i in gt_landmark])
            pre_landmark = torch.tensor(landmark_xy_raw_size).view(1, -1)
            landmark_loss = torch.sum(torch.abs(GT_landmark - pre_landmark)) / len(GT_landmark)
            landmark_losses = landmark_losses + landmark_loss.item()

            gt_pry = line[55:58]
            GT_pry = torch.tensor([float(i) for i in gt_pry])
            pre_pry = (angle_rpy * 10).cpu().view(1, -1)
            pry_loss = torch.sum(torch.abs(GT_pry - pre_pry)) / len(GT_pry)
            pry_losses = pry_losses + pry_loss.item()

            p_loss = torch.abs(GT_pry[0] - pre_pry[0][0])
            r_loss = torch.abs(GT_pry[1] - pre_pry[0][1])
            y_loss = torch.abs(GT_pry[2] - pre_pry[0][2])
            p_losses = p_losses + p_loss.item()
            y_losses = y_losses + y_loss.item()
            r_losses = r_losses + r_loss.item()

            if 0:
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
                    img_raw = draw_landmark_line(img_raw, last_landmark_x, last_landmark_y, (0, 0, 255))

                # save image
                label = 'pre Y  P  R  ' + str('{0:.2f} '.format(float(angle_rpy[0][2]) * 10)) + str(
                    '{0:.2f} '.format(float(angle_rpy[0][0]) * 10)) + str(
                    '{0:.2f} '.format(float(angle_rpy[0][1]) * 10))
                label_gt = 'GT  Y  P  R  ' + str('{0:.2f} '.format(float(gt_pry[2]))) + str(
                    '{0:.2f} '.format(float(gt_pry[0]))) + str(
                    '{0:.2f} '.format(float(gt_pry[1])))
                cv2.putText(img_raw, label, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (1, 111, 255), 2)
                cv2.putText(img_raw, label_gt, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (1, 211, 1), 2)

                print(label)
                cv2.imshow("test", img_raw)
                if cv2.waitKey(1) == 27:
                    break
        f.close()
    #print("avg iou :", ious / len_file)
   # print("avg landmark pixel :", landmark_losses / len_file)
    print("avg angle :", pry_losses / len_file)
    #print("avg p angle :", p_losses / len_file)
    #print("avg r angle :", r_losses / len_file)
    #print("avg y angle :", y_losses / len_file)
