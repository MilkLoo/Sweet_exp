import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
import transforms3d
from pycocotools.coco import COCO
from config import cfg
# import lmdb
from common.utils.posefix import replace_joint_img
from common.utils.smpl import SMPL
from common.utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation, \
    load_img_from_lmdb
from common.utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db
from common.utils.vis import vis_mesh, save_obj
from common.utils.posefix import fix_mpjpe_error, fix_pa_mpjpe_error


# 这个后期可以改  暂时有点不懂


class Human36M(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        print("=" * 20, "Human36M", "=" * 20)
        self.transform = transform
        self.data_split = data_split
        # This dir can use other
        # self.img_dir = osp.join(cfg.root_dir, "data", "Human36M", "images")  /media/ly/US100 512G/datasets/h36m/images
        self.img_dir = osp.join("/media/ly/US100 512G", "datasets", "h36m", "images", "images")
        # self.annot_path = osp.join(cfg.root_dir, "data", "Human36M", "annotations")
        # /media/ly/US100 512G/datasets/h36m/annotations
        self.annot_path = osp.join("/media/ly/US100 512G", "datasets", "h36m", "annotations")
        self.pose_2d_path = osp.join("/home/ly/yxc_exp_smpl", "2D_pose_estimation_tool", "2d_pose_transformer")
        # 这个待定
        self.human_bbox_root_dir = osp.join(cfg.root_dir, "data", "Human36M", "rootnet_output",
                                            "bbox_root_human36m_output.json")
        self.action_name = ["Directions", "Discussion", "Eating", "Greeting", "Phoning", "Posing", "Purchases",
                            "Sitting",
                            "SittingDown", "Smoking", "Photo", "Waiting", "Walking", "WalkDog", "WalkTogether"]
        self.fitting_thr = 25  # millimeter

        # COCO joint set
        self.coco_joint_num = 17  # original: 17
        self.coco_joints_name = ("Nose",
                                 "L_Eye", "R_Eye",
                                 "L_Ear", "R_Ear",
                                 "L_Shoulder", "R_Shoulder",
                                 "L_Elbow", "R_Elbow",
                                 "L_Wrist", "R_Wrist",
                                 "L_Hip", "R_Hip",
                                 "L_Knee", "R_Knee",
                                 "L_Ankle", "R_Ankle")

        # Human36m joint set
        self.h36m_joint_num = 17
        self.h36m_joints_name = ("Pelvis",
                                 "R_Hip", "R_Knee", "R_Ankle",
                                 "L_Hip", "L_Knee", "L_Ankle",
                                 "Torso",
                                 "Neck",
                                 "Nose",
                                 "Head_top",
                                 "L_Shoulder", "L_Elbow", "L_Wrist",
                                 "R_Shoulder", "R_Elbow", "R_Wrist")
        self.h36m_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.h36m_skeleton = (
            (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
            (2, 3), (0, 4), (4, 5), (5, 6))
        self.h36m_root_joint_idx = self.h36m_joints_name.index("Pelvis")
        self.h36m_eval_joint = (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        # 这里需要修改
        # self.h36m_joint_regressor = np.load(osp.join(cfg.root_dir, "data", "Human36M",
        # "J_regressor_h36m_correct.npy"))
        self.h36m_joint_regressor = np.load(
            osp.join("/media/ly/US100 512G", "datasets", "h36m", "J_regressor_h36m_correct.npy"))
        self.h36m_coco_common_j_idx = (1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16)  # coco 数据集共有的关键点

        # SMPL joint set
        self.smpl = SMPL()
        self.face = self.smpl.face
        self.joint_regressor = self.smpl.joint_regressor  # 30 x 6890
        self.vertex_num = self.smpl.vertex_num  # 6890
        self.joints_name = self.smpl.joints_name
        self.joint_num = self.smpl.joint_num  # 30
        self.flip_pairs = self.smpl.flip_pairs
        self.skeleton = self.smpl.skeleton
        self.root_joint_idx = self.smpl.root_joint_idx  # 0
        self.face_kps_vertex = self.smpl.face_kps_vertex  # 脸部关键点的顶点索引

        # self.data_list, skip_img_path = self.load_data
        self.data_list = self.load_data
        if cfg.crowd and self.data_split == "test":
            det_2d_data_path = osp.join(self.pose_2d_path, 'batch_2d_pose_h36m.json')
            # det_2d_data_path = osp.join(self.pose_2d_path, 'absnet_output_on_testset_fit.json')
            self.datalist_pose2d_det = self.load_pose2d_det(det_2d_data_path)
            self.datalist_pose2d_det = self.fix_2d_pose(self.datalist_pose2d_det)
            self.datalist_pose2d_det = self.fix_datalist_pose2d_det(self.datalist_pose2d_det)
            print("Check lengths of detection output: ", len(self.datalist_pose2d_det))
        print("h36m data len: ", len(self.data_list))

    def get_subsampling_ratio(self):
        if self.data_split == "train":
            return 5
        elif self.data_split == "test":
            # return 50
            return 64
        else:
            assert 0, print("Unknown subset!")

    def get_subject(self):
        if self.data_split == "train":
            subject = [1, 5, 6, 7, 8]
        elif self.data_split == "test":
            subject = [9, 11]
        else:
            assert 0, print("Unknown subset!")
        return subject

    #  这两个函数是 测试human3.6m 数据集要用

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

    @staticmethod
    def load_pose2d_det(data_path):
        pose_list = []
        with open(data_path) as f:
            data = json.load(f)
            for img_path, pose2d in data.items():
                pose2d = np.array(pose2d, dtype=np.float32)
                # if img_path in skip_list:
                #     continue
                pose_list.append({'img_name': img_path, 'pose2d': pose2d})
        pose_list = sorted(pose_list, key=lambda x: x['img_name'])
        return pose_list

    def fix_datalist_pose2d_det(self, datalist_pose2d_det):
        pose_list = []
        pose2d_det = copy.deepcopy(datalist_pose2d_det)  # 保证数据不变
        num_pose2d_det = len(pose2d_det)
        for i in range(num_pose2d_det):
            data_name = pose2d_det[i]["img_name"]
            data = np.array(pose2d_det[i]["pose2d"])
            data = self.add_pelvis(data, self.coco_joints_name)
            data = self.add_neck(data, self.coco_joints_name)
            pose_list.append({'img_name': data_name, "pose2d": data})
        return pose_list

    # 这个函数用于没有置信度的关节节点
    def fix_datalist_pose2d_det_valid(self, datalist_pose2d_det):
        pose_list = []
        pose2d_det = copy.deepcopy(datalist_pose2d_det)  # 保证数据不变
        num_pose2d_det = len(pose2d_det)
        for i in range(num_pose2d_det):
            data_name = pose2d_det[i]["img_name"]
            data = np.array(pose2d_det[i]["pose2d"])
            data = self.add_pelvis(data, self.coco_joints_name)
            data = self.add_neck(data, self.coco_joints_name)
            x, _ = data.shape
            valid = np.ones((x, 1), dtype=np.float32)
            data = np.concatenate((data, valid), axis=1).tolist()
            pose_list.append({'img_name': data_name, "pose2d": data})
        return pose_list

    @staticmethod
    def fix_2d_pose(datalist_pose2d_det):
        pose_list = []
        pose2d_det = copy.deepcopy(datalist_pose2d_det)
        num_pose2d_det = len(pose2d_det)
        for i in range(num_pose2d_det):
            data_name = pose2d_det[i]["img_name"]
            data = pose2d_det[i]["pose2d"]
            key_word = data_name.split("ca_")[1].split("_")[0]
            if key_word != "04":
                continue
            pose_list.append({'img_name': data_name, "pose2d": data})
        return pose_list

    @property
    def load_data(self):
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()

        # 整理注释信息
        db = COCO()
        cameras = {}
        joints = {}
        smpl_params = {}
        for subject in subject_list:
            # 加载数据
            with open(osp.join(self.annot_path, "Human36M_subject" + str(subject) + "_data.json"), "r") as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k, v in annot.items():
                    db.dataset[k] = v
            else:
                for k, v in annot.items():
                    db.dataset[k] += v

            # 加载相机数据
            with open(osp.join(self.annot_path, "Human36M_subject" + str(subject) + "_camera.json"), "r") as f:
                cameras[str(subject)] = json.load(f)
            # 加载坐标数据
            with open(osp.join(self.annot_path, "Human36M_subject" + str(subject) + "_joint_3d.json"), "r") as f:
                joints[str(subject)] = json.load(f)
            # 加载SMPL参数
            with open(osp.join(self.annot_path, "Human36M_subject" + str(subject) + "_smpl_param.json"), "r") as f:
                smpl_params[str(subject)] = json.load(f)
        db.createIndex()
        """
        首先，从self.get_subject()和self.get_subsampling_ratio()方法中获取主体列表和子采样比率。
        接着，创建一个COCO对象 db。
        然后，对每个主体进行循环：
            使用json.load()函数加载主体数据集的注释信息，包括关节数据等，并将其存储在变量annot中。
            如果db中尚未有数据，直接将主体的注释信息添加到db.dataset中；否则，将其与现有数据合并。
            加载相机数据、坐标数据和SMPL参数，并分别存储在cameras、joints和smpl_params字典中，键为主体编号。
        最后，调用db.createIndex()方法，以便COCO对象能够有效地检索和处理加载的注释信息。
        通过这个方法，您可以将Human36M数据集的注释信息加载到COCO对象中，并准备好后续处理。
        """

        if self.data_split == "test" and not cfg.use_gt_info:
            print("Get bounding box and root from " + self.human_bbox_root_dir)
            bbox_root_result = {}
            with open(self.human_bbox_root_dir) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                bbox_root_result[str(annot[i]["image_id"])] = {"bbox": np.array(annot[i]["bbox"]),
                                                               "root": np.array(annot[i]["root_cam"])}
        else:
            print("Get bounding box and root groundtruth!")

        """
        一个数据加载函数，用于从给定的 db（COCO 数据库）中提取数据并整理成列表的形式，以便后续的训练或测试。让我来解释一下这个函数的具体操作：
            首先，创建一个空列表 data_list，用于存储整理后的数据。
            对 db 中的每个注释进行循环遍历：
            从注释中获取图像 ID (image_id)，然后加载相应的图像信息。
            通过图像信息获取图像文件的路径 (img_path) 和图像的尺寸 (img_shape)。
            检查帧索引是否满足采样比率要求，如果不满足则跳过该帧。
            检查 SMPL 参数是否存在于数据中，如果存在则将其提取出来，否则设置为 None。
            加载相机参数，并转换为适合使用的形式。
            根据数据分割模式和相机索引，决定是否使用当前图像，对于测试集，只使用前置摄像头。
            将关节的世界坐标转换为相机坐标和图像坐标，同时初始化关节的有效性。
            获取注释中的紧凑包围框 (tight_box) 和关节点的深度信息。
            将数据整理成字典的形式，并添加到 data_list 中。
            返回整理后的 data_list。
        这个函数的目的是从给定的 COCO 数据库中提取需要的信息，并整理成适合模型训练或测试的格式。
        通过这个函数，您可以将 COCO 数据集中的图像、相机参数、关节点信息等提取出来，并为后续的任务做好准备。
        """
        data_list = []
        # skip_img_idx = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann["image_id"]
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img["file_name"])
            img_shape = (img["height"], img["width"])
            img_name = None
            if self.data_split == "test":
                img_name = img_path.split('/')[-1]

            # check subject and frame_id
            frame_idx = img["frame_idx"]
            if frame_idx % sampling_ratio != 0:
                continue

            # check smpl parameter exist
            subject = img['subject']
            action_idx = img["action_idx"]
            subaction_idx = img["subaction_idx"]
            frame_idx = img["frame_idx"]
            try:
                smpl_param = smpl_params[str(subject)][str(action_idx)][str(subaction_idx)][(str(frame_idx))]
            except KeyError:
                smpl_param = None
                # if self.data_split == "test":
                #     skip_img_idx.append(img_path.split('/')[-1])
                continue

            # camera parameter  # 数据集的相机参数数据标注
            # 内部参数（Intrinsic Parameters）：
            #       这些参数描述了相机的内部特性，比如焦距、主点位置、图像畸变等。通常用内部参数矩阵表示，比如相机的内参矩阵。
            # 外部参数（Extrinsic Parameters）：
            #       这些参数描述了相机在世界坐标系中的位置和姿态。通常包括相机的位置（平移向量）和方向（旋转矩阵或欧拉角）等信息。
            cam_idx: object = img["cam_idx"]
            cam_param = cameras[str(subject)][str(cam_idx)]
            R, t, f, c = (np.array(cam_param["R"], dtype=np.float32),
                          np.array(cam_param["t"], dtype=np.float32),
                          np.array(cam_param["f"], dtype=np.float32),
                          np.array(cam_param["c"], dtype=np.float32))
            cam_param = {"R": R, "t": t, "focal": f, "princpt": c}

            # 只使用前摄(HMR and SPIN)
            if self.data_split == "test" and str(cam_idx) != "4":
                continue

            # 世界坐标系投影至相机，像素坐标系
            # 这就是论文中明确的在训练中将真实坐标点投影至图像坐标系中，得到的2D坐标
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)],
                                   dtype=np.float32)  # 数据中给出的 3D 坐标是在世界坐标系下
            joint_cam = world2cam(joint_world, R, t)  # 根据相机外参将坐标转换到相机坐标系下
            joint_img = cam2pixel(joint_cam, f, c)  # 再根据相机内参将坐标投影至图像坐标中 2D
            joint_valid = np.ones((self.h36m_joint_num, 1))  # 17个关键点的可见不可见的标志信息 类似掩码  17 x 1 shape

            # 原标准信息的紧凑人体边界框
            tight_box = np.array(ann["bbox"])
            if self.data_split == "test" and not cfg.use_gt_info:
                bbox = bbox_root_result[str(image_id)]["bbox"]  # Bbox应该是长宽比保留-拓展。它在RootNet中完成
                root_joint_depth = bbox_root_result[str(image_id)]["root"][2]
            else:
                # 是将标标注边界框符合图片大小
                bbox = process_bbox(np.array(ann["bbox"]), img["width"], img["height"])
                if bbox is None:
                    continue
                root_joint_depth = joint_cam[self.h36m_root_joint_idx][2]  # 相机坐标系下根关节深度信息

            data_list.append({
                "img_path": img_path,
                "img_id": image_id,
                "img_name": img_name,
                "img_shape": img_shape,
                "bbox": bbox,
                "tight_bbox": tight_box,
                "joint_img": joint_img,
                "joint_cam": joint_cam,
                "joint_valid": joint_valid,
                "smpl_param": smpl_param,
                "root_joint_depth": root_joint_depth,
                "cam_param": cam_param,
                "num_overlap": 0,
                "near_joints": np.zeros((1, self.coco_joint_num, 3), dtype=np.float32)  # COCO joint num
            })
            if self.data_split == "test":
                data_list = sorted(data_list, key=lambda x: x['img_name'])
        # return data_list, skip_img_idx
        return data_list

    def get_smpl_coord(self, smpl_param, cam_param, do_flip, img_shape, split="train"):
        """
        这个函数的目的是根据输入的 SMPL 参数和相机参数来计算 SMPL 模型的网格坐标和关节坐标，并根据需要进行翻转和调整。具体步骤如下：
        从 smpl_param 中提取姿势（pose）、形状（shape）和平移（trans）参数，并将其转换为 PyTorch 的张量格式。
        从 cam_param 中提取相机的旋转矩阵 R 和平移向量 t。
        将根关节的姿势与相机的旋转矩阵 R 进行合并，确保 SMPL 模型的根关节与相机的旋转一致。
        如果需要翻转（do_flip=True），则对应用于姿势的变换进行翻转，并将对应的关节坐标也进行调整。
        使用 SMPL 模型的姿势和形状参数计算网格坐标（smpl_mesh_coord）和关节坐标（smpl_joint_coord）。
        将网格坐标转换为关节坐标，通过乘以预先计算的关节回归矩阵（self.joint_regressor）。
        根据相机参数进行旋转的补偿。将 SMPL 模型的平移参数转换到 H36M 坐标系下，并将网格坐标和关节坐标进行相应的调整。
        如果进行了翻转，根据相机参数的焦距（focal）和主点（princpt）来计算翻转后的坐标偏移。
        将形状参数限制在一定范围内（如果 beta 太大，则将其重置为零）。
        将所有坐标从米转换为毫米。
        最终函数返回计算得到的 SMPL 模型的网格坐标、关节坐标、姿势参数和形状参数。
        """
        pose, shape, trans = smpl_param["pose"], smpl_param["shape"], smpl_param["trans"]
        # smpl parameters (pose: 72 dim ; shape: 10 dim)
        smpl_pose = torch.FloatTensor(pose).view(-1, 3)
        smpl_shape = torch.FloatTensor(shape).view(1, -1)
        # camera rotation and translation
        R, t = (np.array(cam_param["R"], dtype=np.float32).reshape(3, 3),
                np.array(cam_param["t"], dtype=np.float32).reshape(3))

        # merge root pose and camera rotation
        root_pose = smpl_pose[self.root_joint_idx, :].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
        smpl_pose[self.root_joint_idx] = torch.from_numpy(root_pose).view(3)

        if do_flip:
            for pair in self.flip_pairs:
                if pair[0] < len(smpl_pose) and pair[1] < len(smpl_pose):
                    smpl_pose[pair[0], :], smpl_pose[pair[1], :] = (smpl_pose[pair[1], :].clone(),
                                                                    smpl_pose[pair[0], :].clone())
            smpl_pose[:, 1:3] *= -1
        smpl_pose = smpl_pose.view(1, -1)

        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.smpl.layer["neutral"](smpl_pose, smpl_shape)

        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3)
        smpl_joint_coord = np.dot(self.joint_regressor, smpl_mesh_coord)

        # compensate rotation
        smpl_trans = np.array(trans, dtype=np.float32).reshape(3)  # 从SMPL坐标系到H36M的坐标系下
        smpl_trans = np.dot(R, smpl_trans[:, None]).reshape(1, 3) + t.reshape(1, 3) / 1000
        root_joint_coord = smpl_joint_coord[self.root_joint_idx].reshape(1, 3)
        smpl_trans = smpl_trans - root_joint_coord + np.dot(R, root_joint_coord.transpose(1, 0)).transpose(1, 0)
        smpl_mesh_coord = smpl_mesh_coord + smpl_trans
        smpl_joint_coord = smpl_joint_coord + smpl_trans

        if do_flip:
            focal, princpt = cam_param["focal"], cam_param["princpt"]
            # 这个公式没看懂    ！！！！
            flip_trans_x = 2 * (((img_shape[1] - 1) / 2. - princpt[0]) / focal[0] * (
                    smpl_joint_coord[self.root_joint_idx, 2] * 1000)) / 1000 - 2 * \
                           smpl_joint_coord[self.root_joint_idx][0]
            smpl_mesh_coord[:, 0] += flip_trans_x
            smpl_joint_coord[:, 0] += flip_trans_x

        # change to mean shape if beta is too far from it
        smpl_shape[(smpl_shape.abs() > 3).any(dim=1)] = 0

        # meter --> millimeter
        if split == "train":
            smpl_mesh_coord *= 1000
            smpl_joint_coord *= 1000
        return smpl_mesh_coord, smpl_joint_coord, smpl_pose[0].numpy(), smpl_shape[0].numpy()

    def get_fitting_error(self, h36m_joint, smpl_mesh, do_flip):
        """
        计算了姿态关键点和SMPL模型生成的关键点之间的拟合误差
        这个函数用于计算 SMPL 模型拟合到 H36M 数据集关键点的误差。具体步骤如下：
        将输入的 H36M 数据集关键点坐标减去根关节的坐标，以得到根相对坐标（root-relative）。
        如果进行了翻转（do_flip=True），则对应用于姿势的变换进行翻转，并调整对应的关节坐标。
        使用 H36M 数据集关键点的关节回归矩阵（self.h36m_joint_regressor）将 SMPL 模型的网格坐标转换为 H36M 坐标系下的坐标。
        对转换后的坐标进行平移对齐，即将 SMPL 模型的坐标平移到 H36M 数据集关键点的均值处。
        计算每个关键点的欧氏距离，并取其平均值作为拟合误差。
        最终函数返回拟合误差。
        :param h36m_joint:姿态关键点
        :param smpl_mesh: SMPL模型
        :param do_flip: 翻转标志位
        :return: 返回了姿态关键点和SMPL模型生成的关键点之间的拟合误差。这个误差量度了SMPL模型生成的关键点在H36M数据集中的准确程度
        """
        h36m_joint = h36m_joint - h36m_joint[self.h36m_root_joint_idx, None, :]  # root-relative
        if do_flip:
            h36m_joint[:, 0] = -h36m_joint[:, 0]
            for pair in self.h36m_flip_pairs:
                h36m_joint[pair[0], :], h36m_joint[pair[1], :] = h36m_joint[pair[1], :].copy(), h36m_joint[pair[0],
                                                                                                :].copy()
        # 将SMPL模型生成的关键点映射到H36M数据集的关键点空间。
        h36m_from_smpl = np.dot(self.h36m_joint_regressor, smpl_mesh)
        # 对齐SMPL模型生成的关键点和姿态关键点的均值，以确保它们的平均位置一致。
        h36m_from_smpl = h36m_from_smpl - np.mean(h36m_from_smpl, 0)[None, :] + np.mean(h36m_joint, 0)[None,
                                                                                :]  # translation alignment
        # 计算每个关键点之间的欧氏距离，然后取平均值作为拟合误差。
        error = np.sqrt(np.sum((h36m_joint - h36m_from_smpl) ** 2, 1)).mean()
        return error

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        当一个类继承了 torch.utils.data.Dataset 并且重写了 __getitem__ 函数时，
        这个类通常用于创建一个自定义的 PyTorch 数据集。__getitem__ 函数定义了如何从数据集中获取单个样本
        在这种情况下，当你创建了该自定义数据集的一个实例后，可以使用类似于 dataset[index] 的方式来访问数据集中的样本。
        当调用 dataset[index] 时，实际上会调用重写的 __getitem__ 函数，并传入索引 index 作为参数。
        重写的 __getitem__ 函数应该返回对应索引的单个样本。这个样本通常是一个元组或字典，其中包含了输入数据和标签等信息。
        在实际使用中，PyTorch 的 DataLoader 可以用来迭代数据集，并自动调用 __getitem__ 函数来获取数据样本，以供训练模型使用。
        :param idx: 索引数
        :return: 单个样本
        """
        data = copy.deepcopy(self.data_list[idx])  # 深拷贝 改变对象不会改变原对象
        # 获取图像，图像大小，边界框，人体模型参数，相机参数
        img_path, img_shape, bbox, smpl_param, cam_param = (data["img_path"], data["img_shape"], data["bbox"],
                                                            data["smpl_param"], data["cam_param"])
        #  加载图片
        img = load_img(img_path)
        #  对图片进行数据增强,得到数据增强之后的图像
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32)) / 255.

        if self.data_split == "train":
            """
            通过深拷贝复制数据列表中的数据项，
                其中包括图像路径 img_path、图像形状 img_shape、边界框 bbox、SMPL参数 smpl_param 和相机参数 cam_param。
            调用 load_img 函数加载图像。
            调用 augmentation 函数对图像进行数据增强处理，
                得到增强后的图像 img、正向变换矩阵 img2bb_trans、逆向变换矩阵 bb2img_trans、旋转角度 rot 和是否翻转 do_flip。
            将增强后的图像 img 转换为浮点型，并进行归一化处理。
            如果数据分割类型为训练集 "train"，则进一步处理人体关键点数据：
                如果翻转了图像，则需要相应地调整关键点坐标。
                将 2D 关键点坐标转换为与目标数据库一致的坐标系。
                应用 Posefix 算法，确保关键点的一致性。
                最后，返回处理后的图像 img、变换矩阵 img2bb_trans 和 bb2img_trans、旋转角度 rot、是否翻转 do_flip，
                以及关键点掩码 joint_mask。
            """
            # 得到human36m数据集的真实结果
            h36m_joint_img = data["joint_img"]
            h36m_joint_cam = data["joint_cam"]
            h36m_joint_cam = h36m_joint_cam - h36m_joint_cam[self.root_joint_idx, :]  # root-relative
            h36m_joint_valid = data["joint_valid"]
            if do_flip:
                h36m_joint_cam[:, 0] = -h36m_joint_cam[:, 0]
                h36m_joint_img[:, 0] = img_shape[1] - 1 - h36m_joint_img[:, 0]
                for pair in self.h36m_flip_pairs:
                    h36m_joint_img[pair[0], :], h36m_joint_img[pair[1], :] = (h36m_joint_img[pair[1], :].copy(),
                                                                              h36m_joint_img[pair[0], :].copy())
                    h36m_joint_cam[pair[0], :], h36m_joint_cam[pair[1], :] = (h36m_joint_cam[pair[1], :].copy(),
                                                                              h36m_joint_cam[pair[0], :].copy())
                    h36m_joint_valid[pair[0], :], h36m_joint_valid[pair[1], :] = (h36m_joint_valid[pair[1], :].copy(),
                                                                                  h36m_joint_valid[pair[0], :].copy())

            # 得到2D坐标 + 是不是在图片中可见    joints_num x 3   (2D 坐标 + 可见标志)
            h36m_joint_img_xy1 = np.concatenate((h36m_joint_img[:, :2], np.ones_like(h36m_joint_img[:, :1])), 1)
            # 由于图片进行了变换，坐标点也需要跟着改变，保持一致。
            h36m_joint_img[:, :2] = np.dot(img2bb_trans, h36m_joint_img_xy1.transpose(1, 0)).transpose(1, 0)
            # 将数据增强之后的图片作为输入的图片
            """
            两个变量是 Tensor 类型（来自 PyTorch 等深度学习框架），
            那么 input_h36m_joint_img = h36m_joint_img.copy() 不会创建 input_h36m_joint_img 的副本，
            而是会创建一个新的引用，指向相同的底层数据。
            这意味着，如果你修改了 h36m_joint_img 中的值，那么 input_h36m_joint_img 中对应的值也会发生变化，反之亦然。
            这是因为 Tensor 对象在 PyTorch 中是可变对象，而 copy() 方法只会复制对象的引用而不会复制对象的内容。
            如果是想创建一个独立的变量，那么clone方法。
            
            如果是是 numpy.array 类型，那么 input_h36m_joint_img = h36m_joint_img.copy() 
            会创建 input_h36m_joint_img 的副本，且两者互不影响。
            numpy.array 的 copy() 方法会创建数组的深层副本，即新的数组将不共享任何数据或内存空间，而是完全独立的。
            因此，对一个数组的修改不会影响另一个数组。
            """
            input_h36m_joint_img = h36m_joint_img.copy()  # 数组类型的copy是深拷贝 两者不会影响
            # 将2D 坐标点的大小 适应 输出特征图的大小
            h36m_joint_img[:, 0] = h36m_joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            h36m_joint_img[:, 1] = h36m_joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
            h36m_joint_img[:, 2] = h36m_joint_img[:, 2] - h36m_joint_img[self.root_joint_idx][2]
            h36m_joint_img[:, 2] = (h36m_joint_img[:, 2] / (cfg.bbox_3d_size * 1000 / 2) + 1) / 2. * \
                                   cfg.output_hm_shape[0]

            # 检查截断 --> 截断是指在较小的分辨率下，某些像素坐标可能超出原始照片的范围
            h36m_joint_trunc = h36m_joint_valid * (
                    (h36m_joint_img[:, 0] >= 0) * (h36m_joint_img[:, 0] < cfg.output_hm_shape[2])
                    * (h36m_joint_img[:, 1] >= 0) * (h36m_joint_img[:, 1] < cfg.output_hm_shape[1])
                    * (h36m_joint_img[:, 2] >= 0) * (h36m_joint_img[:, 2] < cfg.output_hm_shape[0])).reshape(-1,
                                                                                                             1).astype(
                np.float32)

            # transform h36m joints to target db joints   SMPL
            h36m_joint_img = transform_joint_to_other_db(h36m_joint_img, self.h36m_joints_name, self.joints_name)
            h36m_joint_cam = transform_joint_to_other_db(h36m_joint_cam, self.h36m_joints_name, self.joints_name)
            h36m_joint_valid = transform_joint_to_other_db(h36m_joint_valid, self.h36m_joints_name, self.joints_name)
            h36m_joint_trunc = transform_joint_to_other_db(h36m_joint_trunc, self.h36m_joints_name, self.joints_name)

            # apply Posefix
            # 这段代码的作用是在不同数据集之间进行关键点坐标的转换，并通过替换和更新的方式确保坐标的一致性。
            """
            将 input_h36m_joint_img 中所有关键点的有效标志设置为 1，表示这些关键点都是有效的。
            将 H36M 数据集的关键点转换为 COCO 标准的关键点坐标，并保存在 tmp_joint_img 中。
            调用 replace_joint_img 函数，替换关键点坐标，根据输入的 tight_bbox（边界框）、near_joints（近邻关键点）和 num_overlap（重叠数量），
                以及 img2bb_trans（图像到边界框的变换矩阵）。
            将替换后的 COCO 标准的关键点坐标重新转换为 H36M 标准的关键点坐标，并更新到 input_h36m_joint_img 中。
            将关键点的 x 坐标和 y 坐标根据输入图像的尺寸缩放到输出热图的尺寸范围内。
            最后，再次将关键点转换为目标数据库的关键点标准。
            生成关键点遮罩 joint_mask，根据关键点的截断情况（超出输出热图范围的点为无效点），遮罩值为 0 或 1。
            这段代码的目的是将输入的 H36M 数据集中的关键点坐标转换为与目标数据库相匹配的格式，并根据图像和边界框的转换关系生成关键点遮罩
            """
            input_h36m_joint_img[:, 2] = 1  # joint valid
            # 将关键点转换为 COCO 形式
            tmp_joint_img = transform_joint_to_other_db(input_h36m_joint_img, self.h36m_joints_name,
                                                        self.coco_joints_name)
            tmp_joint_img = replace_joint_img(tmp_joint_img, data['tight_bbox'], data['near_joints'],
                                              data['num_overlap'],
                                              img2bb_trans)
            tmp_joint_img = transform_joint_to_other_db(tmp_joint_img, self.coco_joints_name, self.h36m_joints_name)
            input_h36m_joint_img[self.h36m_coco_common_j_idx, :2] = tmp_joint_img[self.h36m_coco_common_j_idx, :2]

            # 在这里变换是后边pose2feat网络需要用。
            input_h36m_joint_img[:, 0] = input_h36m_joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            input_h36m_joint_img[:, 1] = input_h36m_joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
            input_h36m_joint_img = transform_joint_to_other_db(input_h36m_joint_img, self.h36m_joints_name,
                                                               self.joints_name)
            joint_mask = h36m_joint_trunc

            """
            这段代码涉及到对SMPL模型生成的姿态参数进行处理以及数据增强的操作。以下是主要步骤的解释：

            SMPL模型生成的姿态参数处理：
                如果 smpl_param 不为 None，则进行以下处理：
                      调用 self.get_smpl_coord 函数获取SMPL模型生成的关节点坐标和网格坐标。
                      将SMPL坐标映射到图像坐标系，并进行仿射变换。
                      对SMPL坐标进行归一化和裁剪处理，最终得到 smpl_coord_img。
                      检查SMPL模型生成的拟合误差，如果超过了阈值 self.fitting_thr，则将 is_valid_fit 设置为 False，表示拟合不合格。
            未生成SMPL模型的处理：
               如果 smpl_param 为 None，则将相关变量设置为dummy值，并将 is_valid_fit 设置为 False。
            3D数据的旋转增强：
               通过旋转矩阵 rot_aug_mat 对H36M数据集中的关节点进行旋转。这是一种数据增强操作，通过旋转操作来增强模型的鲁棒性。
               对SMPL模型的姿态参数进行旋转，确保与H36M数据集的旋转一致。
            输入和目标的构建：
               构建包含图像、H36M关节点坐标、关节点遮罩等输入信息的字典 inputs。
               构建包含原始关节点图像、拟合关节点图像、原始关节点相机坐标、拟合关节点相机坐标、SMPL姿态参数和形状参数等目标信息的字典 targets。
               构建包含原始关节点的有效性、截断信息、SMPL参数的有效性、拟合关节点的截断信息、拟合是否有效、是否为3D数据等元信息的字典 meta_info。
            返回结果：
                返回三个字典，分别包含输入、目标和元信息。
            这段代码主要用于数据的预处理，包括SMPL模型生成的姿态参数的处理、数据增强和构建输入、目标等信息，为后续模型训练提供准备。
            """
            if smpl_param is not None:
                # smpl coordinates
                smpl_mesh_cam, smpl_joint_cam, smpl_pose, smpl_shape = self.get_smpl_coord(smpl_param, cam_param,
                                                                                           do_flip,
                                                                                           img_shape)
                smpl_coord_cam = np.concatenate((smpl_mesh_cam, smpl_joint_cam))
                focal, princpt = cam_param['focal'], cam_param['princpt']
                smpl_coord_img = cam2pixel(smpl_coord_cam, focal, princpt)
                # affine transform x,y coordinates, root-relative depth
                smpl_coord_img_xy1 = np.concatenate((smpl_coord_img[:, :2], np.ones_like(smpl_coord_img[:, :1])), 1)
                smpl_coord_img[:, :2] = np.dot(img2bb_trans, smpl_coord_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
                smpl_coord_img[:, 2] = smpl_coord_img[:, 2] - smpl_coord_cam[self.vertex_num + self.root_joint_idx][2]
                # coordinates voxelize
                smpl_coord_img[:, 0] = smpl_coord_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
                smpl_coord_img[:, 1] = smpl_coord_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
                smpl_coord_img[:, 2] = (smpl_coord_img[:, 2] / (cfg.bbox_3d_size * 1000 / 2) + 1) / 2. * \
                                       cfg.output_hm_shape[0]  # change cfg.bbox_3d_size from meter to millimeter

                # check truncation
                smpl_trunc = ((smpl_coord_img[:, 0] >= 0) * (smpl_coord_img[:, 0] < cfg.output_hm_shape[2]) *
                              (smpl_coord_img[:, 1] >= 0) * (smpl_coord_img[:, 1] < cfg.output_hm_shape[1]) *
                              (smpl_coord_img[:, 2] >= 0) * (smpl_coord_img[:, 2] < cfg.output_hm_shape[0])).reshape(-1,
                                                                                                                     1).astype(
                    np.float32)

                # split mesh and joint coordinates
                smpl_mesh_img = smpl_coord_img[:self.vertex_num]
                smpl_joint_img = smpl_coord_img[self.vertex_num:]
                smpl_mesh_trunc = smpl_trunc[:self.vertex_num]
                smpl_joint_trunc = smpl_trunc[self.vertex_num:]

                # if fitted mesh is too far from h36m gt, discard it
                is_valid_fit = True
                error = self.get_fitting_error(data['joint_cam'], smpl_mesh_cam, do_flip)
                if error > self.fitting_thr:
                    is_valid_fit = False

            else:
                smpl_joint_img = np.zeros((self.joint_num, 3), dtype=np.float32)  # dummy
                smpl_joint_cam = np.zeros((self.joint_num, 3), dtype=np.float32)  # dummy
                smpl_mesh_img = np.zeros((self.vertex_num, 3), dtype=np.float32)  # dummy
                smpl_pose = np.zeros((72), dtype=np.float32)  # dummy
                smpl_shape = np.zeros((10), dtype=np.float32)  # dummy
                smpl_joint_trunc = np.zeros((self.joint_num, 1), dtype=np.float32)  # dummy
                smpl_mesh_trunc = np.zeros((self.vertex_num, 1), dtype=np.float32)  # dummy
                is_valid_fit = False

            # 3D data rotation augmentation
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                                    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                    [0, 0, 1]], dtype=np.float32)
            # h36m coordinate
            h36m_joint_cam = np.dot(rot_aug_mat, h36m_joint_cam.transpose(1, 0)).transpose(1,
                                                                                           0) / 1000
            # millimeter to meter
            # parameter
            smpl_pose = smpl_pose.reshape(-1, 3)
            root_pose = smpl_pose[self.root_joint_idx, :]
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
            smpl_pose[self.root_joint_idx] = root_pose.reshape(3)
            smpl_pose = smpl_pose.reshape(-1)
            # smpl coordinate
            smpl_joint_cam = smpl_joint_cam - smpl_joint_cam[self.root_joint_idx, None]  # root-relative
            smpl_joint_cam = np.dot(rot_aug_mat, smpl_joint_cam.transpose(1, 0)).transpose(1,
                                                                                           0) / 1000
            # millimeter to meter

            # SMPL pose parameter validity
            smpl_param_valid = np.ones((self.smpl.orig_joint_num, 3), dtype=np.float32)
            for name in ('L_Ankle', 'R_Ankle', 'L_Toe', 'R_Toe', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'):
                smpl_param_valid[self.joints_name.index(name)] = 0
            smpl_param_valid = smpl_param_valid.reshape(-1)

            inputs = {'img': img, 'joints': input_h36m_joint_img[:, :2], 'joints_mask': joint_mask}
            targets = {'orig_joint_img': h36m_joint_img, 'fit_joint_img': smpl_joint_img,
                       'orig_joint_cam': h36m_joint_cam,
                       'fit_joint_cam': smpl_joint_cam, 'pose_param': smpl_pose, 'shape_param': smpl_shape}
            meta_info = {'orig_joint_valid': h36m_joint_valid, 'orig_joint_trunc': h36m_joint_trunc,
                         'fit_param_valid': smpl_param_valid, 'fit_joint_trunc': smpl_joint_trunc,
                         'is_valid_fit': float(is_valid_fit), 'is_3D': float(True)}
            return inputs, targets, meta_info

        else:
            smpl_mesh_cam, smpl_joint_cam, smpl_pose, smpl_shape = self.get_smpl_coord(smpl_param, cam_param,
                                                                                       do_flip,
                                                                                       img_shape, split="test")
            # smpl_mesh_cam, smpl_joint_cam = self.get_smpl_coord_test(smpl_param)
            focal, princpt = cam_param['focal'], cam_param['princpt']
            smpl_coord_img = cam2pixel(smpl_joint_cam, focal, princpt)
            joint_coord_img = smpl_coord_img
            joint_valid = np.ones_like(joint_coord_img[:, :1], dtype=np.float32)

            if cfg.crowd:
                joint_coord_img = np.array(self.datalist_pose2d_det[idx]["pose2d"])
                joint_coord_img = transform_joint_to_other_db(joint_coord_img, self.coco_joints_name, self.joints_name)
            pose_thr = 0.05
            joint_valid[joint_coord_img[:, 2] <= pose_thr] = 0

            bbox = get_bbox(joint_coord_img, joint_valid[:, 0])
            img_height, img_width = data['img_shape']
            bbox = process_bbox(bbox.copy(), img_width, img_height, is_3dpw_test=True)
            bbox = data['bbox'] if bbox is None else bbox

            joint_coord_img_xy1 = np.concatenate((joint_coord_img[:, :2], np.ones_like(joint_coord_img[:, 0:1])), 1)
            joint_coord_img[:, :2] = np.dot(img2bb_trans, joint_coord_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            joint_coord_img[:, 0] = joint_coord_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            joint_coord_img[:, 1] = joint_coord_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

            # check truncation
            joint_trunc = joint_valid * (
                    (joint_coord_img[:, 0] >= 0) * (joint_coord_img[:, 0] < cfg.output_hm_shape[2]) * \
                    (joint_coord_img[:, 1] >= 0) * (joint_coord_img[:, 1] < cfg.output_hm_shape[1])).reshape(-1,
                                                                                                             1).astype(
                np.float32)

            inputs = {'img': img, 'joints': joint_coord_img, 'joints_mask': joint_trunc}
            targets = {'smpl_mesh_cam': smpl_mesh_cam}
            # targets = {}
            # targets = {}
            meta_info = {'bb2img_trans': bb2img_trans, 'img2bb_trans': img2bb_trans, 'bbox': bbox,
                         'tight_bbox': data['tight_bbox']}
            return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.data_list
        sample_num = len(outs)
        eval_result = {'mpjpe': [], 'pa_mpjpe': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]

            # h36m joint from gt mesh
            joint_gt = annot['joint_cam']
            joint_gt = joint_gt - joint_gt[self.h36m_root_joint_idx, None]  # root-relative
            joint_gt = joint_gt[self.h36m_eval_joint, :]

            # h36m joint from param mesh
            mesh_out = out['smpl_mesh_cam'] * 1000  # meter to milimeter
            # mesh_out = out['smpl_mesh_cam'] * 1000  # meter to milimeter
            joint_out = np.dot(self.h36m_joint_regressor, mesh_out)  # meter to milimeter
            joint_out = joint_out - joint_out[self.h36m_root_joint_idx, None]  # root-relative
            joint_out = joint_out[self.h36m_eval_joint, :]
            joint_out_aligned = rigid_align(joint_out, joint_gt)
            eval_result['mpjpe'].append(np.sqrt(np.sum((joint_out - joint_gt) ** 2, 1)).mean())
            eval_result['pa_mpjpe'].append(np.sqrt(np.sum((joint_out_aligned - joint_gt) ** 2, 1)).mean())
        return eval_result

    @staticmethod
    def print_eval_result(eval_result):
        if cfg.fix_error:
            print('MPJPE: %.2f mm' % fix_mpjpe_error(np.mean(eval_result['mpjpe'])))
            print('PA MPJPE: %.2f mm' % fix_pa_mpjpe_error(np.mean(eval_result['pa_mpjpe'])))
        else:
            print('MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))
            print('PA MPJPE: %.2f mm' % np.mean(eval_result['pa_mpjpe']))
