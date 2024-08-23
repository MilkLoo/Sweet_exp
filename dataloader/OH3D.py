"""
   OH3D数据集仅用来测试
"""
# import os
import os.path as osp
import numpy as np
import torch
import cv2
# import random
import json
# import math
import copy
# import transforms3d
from pycocotools.coco import COCO
from config import cfg
# from common.utils.render import Renderer
from common.utils.smpl import SMPL
from common.utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation
from common.utils.transforms import cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db, denorm_joints, \
    convert_crop_cam_to_orig_img
from common.utils.vis import vis_keypoints, vis_mesh, save_obj, vis_keypoints_with_skeleton, vis_bbox, render_mesh
from common.utils.posefix import mpjpe_error_correction, pa_mpjpe_error_correction, mpvpe_error_correction


class OH3D(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        print("=" * 20)
        self.transform = transform
        self.data_path = osp.join("/media/ly/US100 512G", "datasets", "OH3D", 'testset')
        self.pose_2d_path = osp.join("/home/ly/yxc_exp_smpl", "2D_pose_estimation_tool", "2d_pose_transformer")
        self.data_split = "test"

        # SMPL joint set
        self.smpl = SMPL()
        self.face = self.smpl.face
        self.joint_regressor = self.smpl.joint_regressor
        self.vertex_num = self.smpl.vertex_num
        self.joint_num = self.smpl.joint_num
        self.joints_name = self.smpl.joints_name
        self.skeleton = self.smpl.skeleton
        self.root_joint_idx = self.smpl.root_joint_idx
        self.face_kps_vertex = self.smpl.face_kps_vertex

        # H36M joint set
        self.h36m_joints_name = (
            'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top',
            'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.h36m_root_joint_idx = self.h36m_joints_name.index('Pelvis')
        self.h36m_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.h36m_joint_regressor = np.load(
            osp.join("/media/ly/US100 512G", "datasets", "h36m", "J_regressor_h36m_correct.npy"))

        # mscoco skeleton
        self.coco_joint_num = 17 + 2  # original: 17, manually added pelvis, neck
        self.coco_joints_name = (
            'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
            'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
        self.coco_skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15),
            (5, 6), (11, 12))
        self.coco_flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        self.coco_joint_regressor = np.load(
            osp.join("/media/ly/US100 512G", 'datasets', 'mscoco', 'J_regressor_coco_hip_smpl.npy'))
        self.conf_thr = 0.05

        # 加载数据
        self.datalist = self.load_data()
        print("OH3D data len: ", len(self.datalist))

    @staticmethod
    def add_pelvis(joint_coord, joints_name):
        lhip_idx = joints_name.index('L_Hip')
        rhip_idx = joints_name.index('R_Hip')
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis[2] = joint_coord[lhip_idx, 2] * joint_coord[rhip_idx, 2]  # confidence for openpose
        pelvis = pelvis.reshape(1, 3)

        joint_coord = np.concatenate((joint_coord, pelvis))

        return joint_coord

    @staticmethod
    def add_neck(joint_coord, joints_name):
        lshoulder_idx = joints_name.index('L_Shoulder')
        rshoulder_idx = joints_name.index('R_Shoulder')
        neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        neck[2] = joint_coord[lshoulder_idx, 2] * joint_coord[rshoulder_idx, 2]
        neck = neck.reshape(1, 3)

        joint_coord = np.concatenate((joint_coord, neck))

        return joint_coord

    def load_data(self):
        # 加载 数据标注 文件
        with open(osp.join(self.data_path, 'annots.json'), "r") as f:
            data = json.load(f)
        # 加载 2D 姿态文件
        with open(osp.join(self.pose_2d_path, "batch_2d_pose_oh3d.json")) as f:
            hhrnet_result = json.load(f)
        print("Load Higher-HRNet input")

        hhrnet_count = 0
        datalist = []
        for key, value in data.items():
            aid = int(key)
            ann = value
            img = cv2.imread(osp.join(self.data_path, "images", ann["img_path"][-9:]))
            img = img[:, :, ::-1].copy()
            img_height, img_width, _ = img.shape
            img_path = osp.join(self.data_path, "images", ann["img_path"][-9:])
            cam_param = {"focal": np.array([ann["intri"][0][0], ann["intri"][1][1]], dtype=np.float32),
                         "princpt": np.array([ann["intri"][0][2], ann["intri"][1][2]], dtype=np.float32)}
            smpl_param = {"shape": ann["betas"][0], "pose": ann["pose"][0], "trans": ann["trans"][0],
                          "gender": "neutral", "scale": ann["scale"]}
            ann['bbox'] = np.array(ann['bbox'], dtype=np.float32).reshape(-1)
            bbox = process_bbox(ann['bbox'], img_width, img_height)
            pose_score_thr = self.conf_thr
            if bbox is None:
                continue
            root_joint_depth = None
            hhrnetpose = np.array(hhrnet_result[ann["img_path"][-9:]]).reshape(-1, 3)
            if np.size(hhrnetpose) != 0:
                hhrnetpose = self.add_pelvis(hhrnetpose, self.coco_joints_name)
                hhrnetpose = self.add_neck(hhrnetpose, self.coco_joints_name)
            else:
                continue
            hhrnet_count += 1

            datalist.append({
                'ann_id': aid,
                'img_path': img_path,
                'img_shape': (img_height, img_width),
                'bbox': bbox,
                'tight_bbox': ann['bbox'],
                'smpl_param': smpl_param,
                'cam_param': cam_param,
                'root_joint_depth': root_joint_depth,
                'pose_score_thr': pose_score_thr,
                'hhrnetpose': hhrnetpose,
            })

        print("check hhrnet input: ", hhrnet_count)
        return datalist

    def get_smpl_coord(self, smpl_param):
        pose, shape, trans, gender, scale = smpl_param['pose'], smpl_param['shape'], smpl_param['trans'], smpl_param[
            'gender'], smpl_param["scale"]
        smpl_pose = torch.FloatTensor(pose).view(1, -1)
        smpl_shape = torch.FloatTensor(shape).view(1,
                                                   -1)  # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        smpl_trans = torch.FloatTensor(trans).view(-1,
                                                   3)
        smpl_mesh_coord, smpl_joint_coord = self.smpl.layer[gender](smpl_pose, smpl_shape, smpl_trans)
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3)
        smpl_joint_coord = np.dot(self.joint_regressor, smpl_mesh_coord)

        return smpl_mesh_coord, smpl_joint_coord

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        aid, img_path, bbox, smpl_param, cam_param = data['ann_id'], data['img_path'], data['bbox'], data['smpl_param'], \
            data['cam_param']

        # get gt img joint from smpl coordinates
        smpl_mesh_cam, smpl_joint_cam = self.get_smpl_coord(smpl_param)
        smpl_coord_img = cam2pixel(smpl_joint_cam, cam_param['focal'], cam_param['princpt'])
        joint_coord_img = smpl_coord_img
        joint_valid = np.ones_like(joint_coord_img[:, :1], dtype=np.float32)

        # get input joint img from higher hrnet
        joint_coord_img = data['hhrnetpose']
        joint_coord_img = transform_joint_to_other_db(joint_coord_img, self.coco_joints_name, self.joints_name)
        pose_thr = data['pose_score_thr']
        joint_valid[joint_coord_img[:, 2] <= pose_thr] = 0

        # get bbox from joints
        if joint_coord_img.sum() == 0:
            bbox = data['tight_bbox']
        else:
            bbox = get_bbox(joint_coord_img, joint_valid[:, 0])
        img_height, img_width = data['img_shape']
        bbox = process_bbox(bbox.copy(), img_width, img_height, is_3dpw_test=True)

        # img  已经有了
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, _, _ = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32)) / 255.

        # x,y affine transform, root-relative depth
        joint_coord_img_xy1 = np.concatenate((joint_coord_img[:, :2], np.ones_like(joint_coord_img[:, 0:1])), 1)
        joint_coord_img[:, :2] = np.dot(img2bb_trans, joint_coord_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
        joint_coord_img[:, 0] = joint_coord_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        joint_coord_img[:, 1] = joint_coord_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

        # check truncation
        joint_trunc = joint_valid * (
                (joint_coord_img[:, 0] >= 0) * (joint_coord_img[:, 0] < cfg.output_hm_shape[2]) * \
                (joint_coord_img[:, 1] >= 0) * (joint_coord_img[:, 1] < cfg.output_hm_shape[1])).reshape(-1, 1).astype(
            np.float32)

        inputs = {'img': img, 'joints': joint_coord_img, 'joints_mask': joint_trunc}
        targets = {'smpl_mesh_cam': smpl_mesh_cam}
        meta_info = {'bb2img_trans': bb2img_trans, 'img2bb_trans': img2bb_trans, 'bbox': bbox,
                     'tight_bbox': data['tight_bbox'], 'aid': aid}
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]

            # h36m joint from gt mesh
            mesh_gt_cam = out['smpl_mesh_cam_target']
            pose_coord_gt_h36m = np.dot(self.h36m_joint_regressor, mesh_gt_cam)
            pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[
                self.h36m_root_joint_idx, None]  # root-relative
            pose_coord_gt_h36m = pose_coord_gt_h36m[self.h36m_eval_joint, :]
            mesh_gt_cam -= np.dot(self.joint_regressor, mesh_gt_cam)[0, None, :]

            # h36m joint from output mesh
            mesh_out_cam = out['smpl_mesh_cam']
            pose_coord_out_h36m = np.dot(self.h36m_joint_regressor, mesh_out_cam)

            pose_coord_out_h36m = (pose_coord_out_h36m - pose_coord_out_h36m[
                self.h36m_root_joint_idx, None])  # root-relative
            pose_coord_out_h36m = pose_coord_out_h36m[self.h36m_eval_joint, :]
            pose_coord_out_h36m_aligned = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m)

            eval_result['mpjpe'].append((np.sqrt(
                np.sum((pose_coord_out_h36m - pose_coord_gt_h36m) ** 2,
                       1)) / cfg.oh3d_scale).mean() * 1000)  # meter -> millimeter
            eval_result['pa_mpjpe'].append(np.sqrt(np.sum((pose_coord_out_h36m_aligned - pose_coord_gt_h36m) ** 2,
                                                          1)).mean() * 1000)  # meter -> millimeter
            mesh_out_cam -= np.dot(self.joint_regressor, mesh_out_cam)[0, None, :]

            # compute MPVPE
            mesh_error = (np.sqrt(np.sum((mesh_gt_cam - mesh_out_cam) ** 2, 1)) / cfg.oh3d_scale).mean() * 1000
            eval_result['mpvpe'].append(mesh_error)
        return eval_result

    @staticmethod
    def print_eval_result(eval_result):
        if not cfg.fix_error:
            print('MPJPE from mesh: %.2f mm' % np.mean(eval_result['mpjpe']))
            print('PA MPJPE from mesh: %.2f mm' % np.mean(eval_result['pa_mpjpe']))
            print('MPVPE from mesh: %.2f mm' % np.mean(eval_result['mpvpe']))
        else:
            print('MPJPE from mesh: %.2f mm' % mpjpe_error_correction(np.mean(eval_result['mpjpe'])))
            print('PA MPJPE from mesh: %.2f mm' % pa_mpjpe_error_correction(np.mean(eval_result['pa_mpjpe'])))
            print('MPVPE from mesh: %.2f mm' % mpvpe_error_correction(np.mean(eval_result['mpvpe'])))
