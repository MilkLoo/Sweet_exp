# import os
# import re
import sys

import cv2
import argparse
import numpy as np
import torch
# import math
# import time
# import scipy
# import matplotlib
# import pylab as plt
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from collections import OrderedDict
# from scipy.ndimage import generate_binary_structure
# from scipy.ndimage import gaussian_filter, maximum_filter

from libary.network.rtpose_vgg import get_model
# from libary.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from libary.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans, human_coord
from libary.utils.paf_to_pose import paf_to_pose_cpp
from libary.config import cfg, update_config

sys.path.append('.')

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='/home/ly/yxc_exp_smpl/2D_pose_estimation_tool/experiments/vgg19_368x368_sgd.yaml',
                    type=str)
parser.add_argument('--weight', type=str,
                    default='/home/ly/yxc_exp_smpl/2D_pose_estimation_tool/model_weight/pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
if __name__ == "__main__":
    args = parser.parse_args()

    # update config file
    update_config(cfg, args)

    model = get_model('vgg19')
    model.load_state_dict(torch.load(args.weight))
    model = torch.nn.DataParallel(model).cuda()
    model.float()
    model.eval()

    test_image = "/home/ly/yxc_exp_smpl/2D_pose_generator_tool/core_code/images/011.png"
    # test_image = "/media/ly/US100 512G/datasets/h36m/images/images/s_09_act_02_subact_01_ca_04/s_09_act_02_subact_01_ca_04_000001.jpg"
    oriImg = cv2.imread(test_image)  # B,G,R order
    shape_dst = np.min(oriImg.shape[0:2])

    # Get results of original image

    with torch.no_grad():
        paf, heatmap, im_scale = get_outputs(oriImg, model, 'rtpose')
    print(im_scale)
    humans = paf_to_pose_cpp(heatmap, paf, cfg)

    out = draw_humans(oriImg, humans)
    cv2.imwrite('picture_output/result.png', out)
