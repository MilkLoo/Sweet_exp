"""
    这个脚本是完成了单张图片的多人、单人的姿态保存成json文件。 --- 尤其注意这是单张的 和  2D 关键点标注器的功能一样。
    单张每次的结果可以自己汇总在 2D_pose_generator_tool / core_code 文件夹下的 2d_pose.json 文件中
    更改一些代码中的图片位置 就可以进行 demo 了。

    后面还会写一个批量的，这个批量需要 文件名称 的 json 文件，自己制作。一般我将批量的用于想要测试的数据集。
"""
import sys
import cv2
import argparse
import numpy as np
import torch
from libary.network.rtpose_vgg import get_model
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from libary.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans, human_coord, \
    get_coco_coord
from libary.utils.paf_to_pose import paf_to_pose_cpp
from libary.config import cfg, update_config
from trans_tool import trans_my_project_format_list, trans_my_demo_format_dict
import json

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

    test_image = "/home/ly/yxc_exp_smpl/2D_pose_generator_tool/core_code/images/075.jpg"
    # test_image = "/media/ly/US100 512G/datasets/h36m/images/images/s_09_act_03_subact_02_ca_02/s_09_act_03_subact_02_ca_02_003651.jpg"
    oriImg = cv2.imread(test_image)  # B,G,R order
    shape_dst = np.min(oriImg.shape[0:2])

    # Get results of original image

    with torch.no_grad():
        paf, heatmap, im_scale = get_outputs(oriImg, model, 'rtpose')
    humans = paf_to_pose_cpp(heatmap, paf, cfg)
    human_coord_info, human_num = human_coord(oriImg, humans)
    all_human_coco_coord = get_coco_coord(human_coord_info, human_num)
    # 以下都没问题了
    my_format = trans_my_project_format_list(all_human_coco_coord)
    end_result = trans_my_demo_format_dict(my_format, "075.jpg")
    json_file_path = "2d_pose_est.json"
    with open(json_file_path, 'w') as json_file:
        json.dump(end_result, json_file)
    print("Json文件保存成功!")
