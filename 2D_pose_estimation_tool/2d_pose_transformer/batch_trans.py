"""
这里写一个Human3.6M的 2d 测试代码
后面有自己的图片数据集的json标注文件，自己参考着写
"""
import os.path as osp
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
from trans_tool import trans_my_demo_format_dict, trans_my_project_format_list_batch
import json
import copy

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


# class ImageData(object):
#     def __init__(self):
#         self.annot_path = osp.join("/media/ly/US100 512G", "datasets", "h36m", "annotations")
#         self.subject_list = [9, 11]
#         self.get_subsampling_ratio = 64
#         self.data = {}
#         self.count = 1
#
#     def read_json(self):
#         for subject in self.subject_list:
#             with open(osp.join(self.annot_path, "Human36M_subject" + str(subject) + "_data.json"), "r") as f:
#                 annot = json.load(f)
#                 for i in range(len(annot["images"])):
#                     img_info = annot["images"][i]["file_name"]  # str
#                     frame_idx = annot["images"][i]["frame_idx"]  # int
#                     if frame_idx % self.get_subsampling_ratio != 0:
#                         continue
#                     self.data[self.count] = img_info
#                     self.count += 1
#         return self.data


# class GeneralImageData(object):
#     def __init__(self):
#         # 这里要写你图片的标注文件地址
#         self.annot_path = ""
#
#     def read_json(self):
#         with open(osp.join(self.annot_path, "具体的json文件"), "r") as f:
#             # 最好 就是 {文件id：图片名称}
#             annot = json.load(f)
#         return annot


if __name__ == "__main__":
    args = parser.parse_args()

    # update config file
    update_config(cfg, args)

    model = get_model('vgg19')
    model.load_state_dict(torch.load(args.weight))
    model = torch.nn.DataParallel(model).cuda()
    model.float()
    model.eval()

    # 要是改的话 自己看着改 就行
    end_result = {}

    # json_file_path = "batch_2d_pose_h36m.json"
    # json_file_path = "batch_2d_pose_oh3d.json"
    # json_file_path = "batch_2d_pose_walk.json"
    # json_file_path = "batch_2d_pose_gripper.json"
    json_file_path = "batch_2d_pose_walk_a.json"
    # batch_img_path = osp.join("/media/ly/US100 512G", "datasets", "h36m", "images", "images")
    # batch_img_path = osp.join("/media/ly/US100 512G", "datasets", "OH3D", 'testset', 'images')
    batch_img_path = osp.join("/home/ly/yxc_exp_smpl/image_output", "walk_a")
    # img_data = ImageData().read_json()
    # img_data_deep = copy.deepcopy(img_data)
    # json_file = "/media/ly/US100 512G/datasets/OH3D/testset/annots.json"
    # json_file = "/media/ly/US100 512G/datasets/OH3D/testset/annots.json"
    # json_file = "/home/ly/yxc_exp_smpl/image_output/Json/gripper.json"
    json_file = "/home/ly/yxc_exp_smpl/image_output/Json/walk_a.json"
    with open(json_file, "r") as f:
        data = json.load(f)

    # for key, value in img_data_deep.items():
    #     image_name_str = value.split("/")[1]
    for key, value in data.items():
        # image_name_str = key + ".jpg"
        image_name_str = key
        test_image = osp.join(batch_img_path, image_name_str)
        oriImg = cv2.imread(test_image)  # B,G,R order
        shape_dst = np.min(oriImg.shape[0:2])
        # Get results of original image
        with torch.no_grad():
            paf, heatmap, im_scale = get_outputs(oriImg, model, 'rtpose')
        humans = paf_to_pose_cpp(heatmap, paf, cfg)
        human_coord_info, human_num = human_coord(oriImg, humans)
        all_human_coco_coord = get_coco_coord(human_coord_info, human_num)
        my_format = trans_my_project_format_list_batch(all_human_coco_coord)
        end_result[image_name_str] = my_format
        end_result[key] = [my_format]
    with open(json_file_path, 'w') as json_file:
        json.dump(end_result, json_file)
    print("Json文件保存成功!")
