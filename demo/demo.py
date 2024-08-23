import sys
# import os
import os.path as osp
import argparse
import numpy as np
import cv2
import colorsys
import json
# import random
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

# import matplotlib.pyplot as plt

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from common.utils.preprocessing import process_bbox, generate_patch_image, get_bbox
from common.utils.transforms import pixel2cam, cam2pixel, transform_joint_to_other_db
from common.utils.vis import vis_mesh, save_obj, render_mesh, vis_coco_skeleton

sys.path.insert(0, cfg.smpl_path)
from common.utils.smpl import SMPL


def add_pelvis(joint_coord, joints_name):
    """
    这段代码实现了一个函数 add_pelvis(joint_coord, joints_name)，用于向关键点坐标数组中添加骨盆（pelvis）关键点。
    具体来说，这个函数的实现如下：
        根据关键点名称列表 joints_name 找到左髋关键点（'L_Hip'）和右髋关键点（'R_Hip'）的索引。
        计算左髋关键点和右髋关键点的坐标的平均值，作为骨盆关键点的坐标。
        同时，将骨盆关键点的置信度设为左髋关键点和右髋关键点置信度的乘积（这里假设关键点的置信度存在于 z 坐标中）。
        将骨盆关键点的坐标转换为形状为 (1, 3) 的数组。
        使用 np.concatenate() 函数将骨盆关键点添加到关键点坐标数组 joint_coord 中。
        返回添加了骨盆关键点后的关键点坐标数组。
        这个函数通常用于在处理人体关键点时，将左和右髋关键点的坐标平均值作为骨盆关键点的估计值，并将其添加到关键点坐标数组中。
        这样可以更完整地表示人体的姿势
    :param joint_coord:关键点坐标数组
    :param joints_name:关键点名称列表
    :return:添加了骨盆关键点后的关键点坐标数组
    """
    lhip_idx = joints_name.index('L_Hip')
    rhip_idx = joints_name.index('R_Hip')
    pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
    pelvis[2] = joint_coord[lhip_idx, 2] * joint_coord[rhip_idx, 2]  # confidence for openpose
    pelvis = pelvis.reshape(1, 3)

    joint_coord = np.concatenate((joint_coord, pelvis))

    return joint_coord


def add_neck(joint_coord, joints_name):
    """
    这段代码实现了一个函数 add_neck(joint_coord, joints_name)，用于向关键点坐标数组中添加颈部（neck）关键点。
    具体来说，这个函数的实现如下：
        根据关键点名称列表 joints_name 找到左肩关键点（'L_Shoulder'）和右肩关键点（'R_Shoulder'）的索引。
        计算左肩关键点和右肩关键点的坐标的平均值，作为颈部关键点的坐标。
        同时，将颈部关键点的置信度设为左肩关键点和右肩关键点置信度的乘积（这里假设关键点的置信度存在于 z 坐标中）。
        将颈部关键点的坐标转换为形状为 (1, 3) 的数组。
        使用 np.concatenate() 函数将颈部关键点添加到关键点坐标数组 joint_coord 中。
        返回添加了颈部关键点后的关键点坐标数组。
    这个函数通常用于在处理人体关键点时，将左右肩关键点的坐标平均值作为颈部关键点的估计值，并将其添加到关键点坐标数组中。
    这样可以更完整地表示人体的姿势。
    :param joint_coord:关键点坐标数组
    :param joints_name:关键点名称列表
    :return:添加了颈部关键点后的关键点坐标数组
    """
    lshoulder_idx = joints_name.index('L_Shoulder')
    rshoulder_idx = joints_name.index('R_Shoulder')
    neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
    neck[2] = joint_coord[lshoulder_idx, 2] * joint_coord[rshoulder_idx, 2]
    neck = neck.reshape(1, 3)

    joint_coord = np.concatenate((joint_coord, neck))

    return joint_coord


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="0", type=str, dest='gpu_ids')
    # 这里改模型权重文件，不同的实验模型不一样。做完实验之后会在 output 文件下的 README.md 文件进行说明。
    parser.add_argument('--model_path', type=str,
                        default='/home/ly/yxc_exp_smpl/output/exp_3/exp_04-29_18:43/checkpoint/snapshot_14.pth.tar')
    parser.add_argument('--img_idx', type=str, default='075')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


if __name__ == "__main__":
    # argument parsing
    args = parse_args()
    cfg.set_args(args.gpu_ids, is_test=True)
    cfg.render = True
    # CuDNN 是 NVIDIA 提供的深度学习库，用于在 NVIDIA GPU 上加速深度学习计算。
    # cudnn.benchmark 参数的作用是告诉 PyTorch 是否使用 CuDNN 的自动调整算法来提高性能。
    cudnn.benchmark = True

    # SMPL joint set
    joint_num = 30  # original: 24. manually add nose, L/R eye, L/R ear, head top
    joints_name = (
        'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe',
        'R_Toe',
        'Neck', 'L_Thorax', 'R_Thorax',
        'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose',
        'L_Eye',
        'R_Eye', 'L_Ear', 'R_Ear', 'Head_top')
    flip_pairs = (
        (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))
    skeleton = (
        (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
        (17, 19),
        (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 24), (24, 15), (24, 25),
        (24, 26),
        (25, 27), (26, 28), (24, 29))

    # SMPl mesh
    vertex_num = 6890
    smpl = SMPL()
    face = smpl.face
    joint_regressor = smpl.joint_regressor
    joint_3d_list = []
    joint_3d_dict = {}

    # other joint set
    coco_joints_name = (
        'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
        'R_Wrist',
        'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
    coco_skeleton = (
        (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15),
        (5, 6),
        (11, 17), (12, 17), (17, 18))

    vis_joints_name = (
        'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
        'R_Wrist',
        'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Thorax', 'Pelvis')
    vis_skeleton = (
        (0, 1), (0, 2), (2, 4), (1, 3), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 17), (6, 17),
        (11, 18),
        (12, 18), (17, 18), (17, 0), (6, 8), (8, 10),)

    # snapshot load  加载好模型
    model_path = args.model_path
    assert osp.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    model = get_model(vertex_num, joint_num, 'test')

    # 用于加载模型并将其移动到 GPU 上进行推理
    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['network'], strict=True)
    model.eval()

    # prepare input image 读取注释好的文件，这里的文件是用2D_pose_generator_tool生成的。
    transform = transforms.ToTensor()
    # pose2d_result_path = './input/2d_pose_result.json'
    pose2d_result_path = '/home/ly/yxc_exp_smpl/2D_pose_generator_tool/core_code/2d_pose.json'
    # pose2d_result_path = "/home/ly/yxc_exp_smpl/2D_pose_estimation_tool/2d_pose_transformer/2d_pose_est.json"
    with open(pose2d_result_path) as f:
        pose2d_result = json.load(f)

    # img_dir = './input/images'
    img_dir = '/home/ly/yxc_exp_smpl/2D_pose_generator_tool/core_code/images'
    # img_dir = "/media/ly/US100 512G/datasets/h36m/images/images/s_09_act_02_subact_01_ca_04"
    # 利用 sorted 函数对结果的键值进行排序，然后遍历排序后的结果
    for img_name in sorted(pose2d_result.keys()):
        img_path = osp.join(img_dir, img_name)
        # 读取到原始图片
        original_img = cv2.imread(img_path)
        # 进行复制两次
        input = original_img.copy()
        input2 = original_img.copy()
        # 得到原始图片的高，宽
        original_img_height, original_img_width = original_img.shape[:2]
        # 得到关键点信息
        coco_joint_list = pose2d_result[img_name]

        # 如果这个图片不在，就进行下一轮循环
        if args.img_idx not in img_name:
            continue

        drawn_joints = []
        c = coco_joint_list
        # manually assign the order of output meshes
        # coco_joint_list = [c[2], c[0], c[1], c[4], c[3]]

        for idx in range(len(coco_joint_list)):
            """ 2D pose input setting & hard-coding for filtering """
            pose_thr = 0.1
            coco_joint_img = np.asarray(coco_joint_list[idx])[:, :3]
            coco_joint_img = add_pelvis(coco_joint_img, coco_joints_name)
            coco_joint_img = add_neck(coco_joint_img, coco_joints_name)
            coco_joint_valid = (coco_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32)

            # filter inaccurate inputs
            """
            det_score = sum(coco_joint_img[:, 2]): 这行代码计算了检测到的人体关键点的置信度得分，通过对所有关键点的置信度进行求和。
                                            这里假设 coco_joint_img 是一个二维数组，每行代表一个人体关键点，第三列是置信度得分。
            if det_score < 1.0: continue: 如果所有检测到的关键点的置信度得分之和小于 1.0，那么跳过当前循环，继续处理下一个人体关键点。
                                            这个条件可能用于过滤掉置信度较低的关键点，只保留相对置信度较高的关键点。
            if len(coco_joint_img[:, 2:].nonzero()[0]) < 1: continue: 这行代码用于进一步过滤掉检测到的关键点数量较少的情况。
            它先通过 coco_joint_img[:, 2:].nonzero() 找到所有置信度不为零的关键点，然后通过 len() 函数计算关键点的数量。
            如果关键点数量小于 1（即没有关键点检测到），则跳过当前循环，继续处理下一个人体关键点。
            """
            det_score = sum(coco_joint_img[:, 2])
            if det_score < 1.0:
                continue
            if len(coco_joint_img[:, 2:].nonzero()[0]) < 1:
                continue
            # filter the same targets
            tmp_joint_img = coco_joint_img.copy()
            continue_check = False
            for ddx in range(len(drawn_joints)):
                drawn_joint_img = drawn_joints[ddx]
                drawn_joint_val = (drawn_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32)
                diff = np.abs(tmp_joint_img[:, :2] - drawn_joint_img[:, :2]) * coco_joint_valid * drawn_joint_val
                diff = diff[diff != 0]
                if diff.size == 0:
                    continue_check = True
                elif diff.mean() < 20:
                    continue_check = True
            if continue_check:
                continue
            drawn_joints.append(tmp_joint_img)

            """ Prepare model input """
            # prepare bbox
            bbox = get_bbox(coco_joint_img, coco_joint_valid[:, 0])  # xmin, ymin, width, height
            bbox = process_bbox(bbox, original_img_width, original_img_height)
            if bbox is None:
                continue
            img, img2bb_trans, bb2img_trans = generate_patch_image(input2[:, :, ::-1], bbox, 1.0, 0.0, False,
                                                                   cfg.input_img_shape)
            img = transform(img.astype(np.float32)) / 255
            img = img.cuda()[None, :, :, :]

            coco_joint_img_xy1 = np.concatenate((coco_joint_img[:, :2], np.ones_like(coco_joint_img[:, :1])), 1)
            coco_joint_img[:, :2] = np.dot(img2bb_trans, coco_joint_img_xy1.transpose(1, 0)).transpose(1, 0)
            coco_joint_img[:, 0] = coco_joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            coco_joint_img[:, 1] = coco_joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

            coco_joint_img = transform_joint_to_other_db(coco_joint_img, coco_joints_name, joints_name)
            coco_joint_valid = transform_joint_to_other_db(coco_joint_valid, coco_joints_name, joints_name)
            coco_joint_valid[coco_joint_img[:, 2] <= pose_thr] = 0

            # check truncation
            coco_joint_trunc = coco_joint_valid * (
                    (coco_joint_img[:, 0] >= 0) * (coco_joint_img[:, 0] < cfg.output_hm_shape[2]) * (
                    coco_joint_img[:, 1] >= 0) * (coco_joint_img[:, 1] < cfg.output_hm_shape[1])).reshape(
                -1, 1).astype(np.float32)
            coco_joint_img, coco_joint_trunc, bbox = torch.from_numpy(coco_joint_img).cuda()[None, :,
                                                     :], torch.from_numpy(
                coco_joint_trunc).cuda()[None, :, :], torch.from_numpy(bbox).cuda()[None, :]

            """ Model forward """
            inputs = {'img': img, 'joints': coco_joint_img, 'joints_mask': coco_joint_trunc}
            targets = {}
            meta_info = {'bbox': bbox}
            with torch.no_grad():
                out = model(inputs, targets, meta_info, 'test')

            # draw output mesh
            # mesh_cam_render = out['mesh_cam_render'][0].cpu().numpy()
            mesh_cam_render = out['mesh_cam_render'][0].cpu().numpy()
            bbox = out['bbox'][0].cpu().numpy()
            princpt = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
            # original_img = vis_bbox(original_img, bbox, alpha=1)  # for debug

            # generate random color
            color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
            # color = colorsys.hsv_to_rgb(0.5, 0.5, 1.0)
            original_img = render_mesh(original_img, mesh_cam_render, face, {'focal': cfg.focal, 'princpt': princpt},
                                       color=color)

            # Save output mesh
            output_dir = 'output'
            file_name = f'{output_dir}/{img_path.split("/")[-1][:-4]}_{idx}.jpg'
            print("file name: ", file_name)
            save_obj(mesh_cam_render, face, file_name=f'{output_dir}/{img_path.split("/")[-1][:-4]}_{idx}.obj')
            cv2.imwrite(file_name, original_img)

            # Draw input 2d pose
            tmp_joint_img[-1], tmp_joint_img[-2] = tmp_joint_img[-2].copy(), tmp_joint_img[-1].copy()
            input = vis_coco_skeleton(input, tmp_joint_img.T, vis_skeleton)
            cv2.imwrite(file_name[:-4] + '_2dpose.jpg', input)

            # Draw 3d joint
            joint_3d = np.dot(joint_regressor, mesh_cam_render).tolist()
            joint_3d_list.append(joint_3d)

    people_num = len(joint_3d_list)
    for i in range(people_num):
        joint_3d_dict[f"people_{i}"] = joint_3d_list[i]
    file_path = "/home/ly/yxc_exp_smpl/tool/3d_keypoint_data/3d_joint_data.json"
    with open(file_path, "w") as json_files:
        json.dump(joint_3d_dict, json_files)
    print("JSON 文件保存成功:", file_path)
