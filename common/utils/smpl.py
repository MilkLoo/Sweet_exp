import numpy as np
# import torch
import os.path as osp
# import json
from config import cfg
from common.utils.smplpytorch.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from common.utils.transforms import build_adj, normalize_adj  # transform_joint_to_other_db
import sys

sys.path.insert(0, cfg.smpl_path)


class SMPL(object):
    """
    这段代码定义了一个 SMPL 类，该类用于处理和操作 SMPL 模型相关的信息。以下是代码的主要功能和属性：
        构造函数 (__init__)
            初始化了三个 SMPL_Layer 的实例，分别对应中性性别(neutral)、男性(male)和女性(female)。每个实例的初始化路径基于配置中的 smpl_path。
            通过调用 get_layer 方法，创建了一个字典 layer，其中包含不同性别的 SMPL_Layer 实例。
            定义了一些 关于 SMPL 模型的属性，例如顶点数 (vertex_num)、面 (face)、关节数 (joint_num) 等。
            增加了额外的关节（例如头部的各个部位，左右眼、耳朵等），并更新了关节的数量。
            get_graph_adj 方法
                    调用 build_adj 和 normalize_adj 函数构建图的邻接矩阵 (graph_adj)。这个图描述了 15 个关键关节的拓扑结构。
            reduce_joint_set 方法
                    对输入的关节坐标进行减少操作，返回仅包含 15 个关节的坐标。
        其他属性和方法
            定义了一些关于骨架 (skeleton)、关节名称 (joints_name)、关节翻转对 (flip_pairs) 等的信息，用于关节和骨架的可视化等应用。
    总体而言，这个 SMPL 类封装了 SMPL 模型的不同性别实例以及与模型相关的信息和操作，
    提供了方便的接口来获取不同性别的 SMPL_Layer 实例，并进行一些关节信息的处理。
    """

    def __init__(self):
        # 文件夹的路径都需要重新查看修改
        self.neutral = SMPL_Layer(gender="neutral", model_root=cfg.smpl_path + "/smplpytorch/native/models")
        self.male = SMPL_Layer(gender="male", model_root=cfg.smpl_path + "/smplpytorch/native/models")
        self.female = SMPL_Layer(gender="female", model_root=cfg.smpl_path + "/smplpytorch/native/models")
        self.layer = {"neutral": self.get_layer(), "male": self.get_layer("male"), "female": self.get_layer("female")}
        self.vertex_num = 6890
        # 顶点拓扑关系
        self.face = self.layer["neutral"].th_faces.numpy()
        self.joint_regressor = self.layer["neutral"].th_J_regressor.numpy()
        self.shape_param_dim = 10
        # v_poser 需要了解  姿势先验
        self.v_poser_code_dim = 32

        # add nose, L/R eye, L/R ear
        self.face_kps_vertex = (331, 2802, 6262, 3489, 3990)  # mesh vertex idx
        nose_onehot = np.array([1 if i == 331 else 0 for i in range(self.joint_regressor.shape[1])],
                               dtype=np.float32).reshape(1, -1)  # 1 x 6890 shape
        left_eye_onehot = np.array([1 if i == 2802 else 0 for i in range(self.joint_regressor.shape[1])],
                                   dtype=np.float32).reshape(1, -1)
        right_eye_onehot = np.array([1 if i == 6262 else 0 for i in range(self.joint_regressor.shape[1])],
                                    dtype=np.float32).reshape(1, -1)
        left_ear_onehot = np.array([1 if i == 3489 else 0 for i in range(self.joint_regressor.shape[1])],
                                   dtype=np.float32).reshape(1, -1)
        right_ear_onehot = np.array([1 if i == 3990 else 0 for i in range(self.joint_regressor.shape[1])],
                                    dtype=np.float32).reshape(1, -1)
        self.joint_regressor = np.concatenate(
            (self.joint_regressor, nose_onehot, left_eye_onehot, right_eye_onehot, left_ear_onehot, right_ear_onehot))
        # add head top
        # self.joint_regressor_extra = np.load(osp.join(cfg.root_dir, "data", "J_regressor_extra.npy"))
        # /media/ly/US100 512G/datasets
        self.joint_regressor_extra = np.load(osp.join("/media/ly/US100 512G", "datasets", "J_regressor_extra.npy"))
        self.joint_regressor = np.concatenate((self.joint_regressor, self.joint_regressor_extra[3:4, :])).astype(
            np.float32)

        self.orig_joint_num = 24
        # 最原始的smpl的关键点 没有鼻子 左右眼  左 右耳  头顶 所以填在其中 作为不同数据集中关键点的超集
        self.joint_num = 30  # original: 24, manually add nose, L/R eyes, L/R ears, head top
        self.six_locate_center = ("Pelvis", "l_Hip", "R_Hip", "L_Shoulder", "R_Shoulder", "Neck")
        self.joints_name = ("Pelvis", "l_Hip", "R_Hip",
                            "Torso", "L_Knee", "R_Knee",
                            "Spine", "L_Ankle", "R_Ankle",
                            "Chest", "L_Toe", "R_Toe",
                            "Neck", "L_Thorax", "R_Thorax",
                            "Head", "L_Shoulder", "R_Shoulder",
                            "L_Elbow", "R_Elbow",
                            "L_Wrist", "R_Wrist",
                            "L_Hand", "R_Hand",
                            "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "Head_top")
        self.flip_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17),
                           (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))
        self.skeleton = ((0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8),
                         (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17), (17, 19),
                         (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20),
                         (20, 22), (9, 12), (12, 24), (24, 15), (24, 25), (24, 26),
                         (25, 27), (26, 28), (24, 29))
        self.root_joint_idx = self.joints_name.index("Pelvis")

        # joint set for PositionNet prediction
        # 作为图网络的拓扑   模型拓扑
        self.graph_joint_num = 15
        self.graph_joint_name = ("Pelvis", "l_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
                                 "Neck", "Head_top", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
                                 "L_Wrist", "R_Wrist")
        self.graph_flip_pairs = ((1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14))
        self.graph_skeleton = ((0, 1), (1, 3), (3, 5),
                               (0, 2), (2, 4), (4, 6),
                               (0, 7), (7, 8), (7, 9), (9, 11), (11, 13),
                               (7, 10), (10, 12), (12, 14))
        # build adj matrix
        self.graph_adj = self.get_graph_adj()

        self.idx_list_15 = []
        for name in self.graph_joint_name:
            idx = self.joints_name.index(name)
            self.idx_list_15.append(idx)
        self.idx_list_6 = []
        for name in self.six_locate_center:
            idx = self.joints_name.index(name)
            self.idx_list_6.append(idx)

    def get_layer(self, gender="neutral"):
        """
        这里得到的是smpl层的路径
        :param gender: 人的性别
        :return: 性别不同，构建的SMPL层是根据性别来的
        """
        if gender == "neutral":
            return self.neutral
        elif gender == "male":
            return self.male
        elif gender == "female":
            return self.female
        else:
            raise ValueError("Gender invalid input:" + gender)

    def get_graph_adj(self):
        adj_mat = build_adj(self.graph_joint_num, self.graph_skeleton, self.graph_flip_pairs)
        normalized_adj = normalize_adj(adj_mat)
        return normalized_adj

    def reduce_joint_set(self, joint):
        return joint[:, self.idx_list_15, :].contiguous()
