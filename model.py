"""
研究内容1: Model: ResNet-50 + Module
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from common.net.resnet import ResNetBackbone
from common.net.module import Pose2Feat, PositionNet, RotationNet, Vposer
from common.net.loss import CoordLoss, ParamLoss, NormalVectorLoss, EdgeLengthLoss
from common.utils.smpl import SMPL
from common.utils.mano import MANO
from config import cfg
from contextlib import nullcontext
import math
from lib.module import MidAttention
from common.utils.transforms import rot6d_to_axis_angle


class Model(nn.Module):
    def __init__(self, backbone, pose2feat, position_net, rotation_net, vposer, module):  # module
        super(Model, self).__init__()
        self.backbone = backbone
        self.pose2feat = pose2feat
        self.position_net = position_net
        self.rotation_net = rotation_net
        self.vposer = vposer
        self.module = module

        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.human_model = MANO()
            self.human_model_layer = self.human_model.layer.cuda()
        else:
            self.human_model = SMPL()
            self.human_model_layer = self.human_model.layer['neutral'].cuda()
        self.root_joint_idx = self.human_model.root_joint_idx
        self.mesh_face = self.human_model.face
        self.joint_regressor = self.human_model.joint_regressor

        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()

    @staticmethod
    def get_camera_trans(cam_param, meta_info, is_render):
        """
        方法用于获取相机的平移参数
        提取相机参数的平移分量和缩放参数：
            cam_param 是一个形状为 (N, 3) 的张量，其中 N 是样本数量。该张量包含了相机参数，其中前两列是平移分量 (x, y)，第三列是缩放参数 (gamma)。
            cam_param[:, :2] 提取了前两列，即相机的 x 和 y 平移分量。
            torch.sigmoid(cam_param[:, 2]) 对第三列应用 Sigmoid 函数，将缩放参数限制在 (0, 1) 范围内，确保其取值为正数。
        计算相机的深度平移 t_z：
            k_value 是一个常量，用于计算深度平移。它基于配置参数 cfg.focal、cfg.camera_3d_size 和 cfg.input_img_shape 计算得出。
            如果 is_render 为 True，则从元信息 meta_info 中提取边界框信息 bbox，并根据渲染结果调整 k_value。
            最后，深度平移 t_z 是根据 k_value 和缩放参数 gamma 计算得出。
        构造相机的平移矩阵 cam_trans：
            torch.cat 函数将提取的平移分量 t_xy 和计算得到的深度平移 t_z 拼接成一个形状为 (N, 3) 的张量，其中 N 是样本数量。
            t_z[:, None] 将深度平移张量 t_z 转换为列向量，以便与平移分量 t_xy 拼接在一起。
        返回相机的平移参数 cam_trans
        :param cam_param: 相机参数
        :param meta_info: 元信息
        :param is_render: 是否渲染的标志位
        :return: 相机的平移参数
        """
        # camera translation
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])  # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0] * cfg.focal[1] * cfg.camera_3d_size * cfg.camera_3d_size / (
                cfg.input_img_shape[0] * cfg.input_img_shape[1]))]).cuda().view(-1)
        if is_render:
            bbox = meta_info['bbox']
            k_value = k_value * math.sqrt(cfg.input_img_shape[0] * cfg.input_img_shape[1]) / (
                    bbox[:, 2] * bbox[:, 3]).sqrt()
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)
        return cam_trans

    @staticmethod
    def make_2d_gaussian_heatmap(joint_coord_img):
        """
        函数用于生成二维高斯热图（2D Gaussian Heatmap），用于表示关键点在图像上的分布情况
        首先，通过 torch.arange 函数创建了横轴和纵轴的坐标网格，
            范围为输出热图的宽度和高度（cfg.output_hm_shape[2] 和 cfg.output_hm_shape[1]）。
            使用 torch.meshgrid 函数创建了二维坐标网格，得到了横轴和纵轴上的坐标矩阵 xx 和 yy。
        将输入的关键点图像坐标 joint_coord_img 与 xx 和 yy 进行差值计算，得到了横轴和纵轴上的差值矩阵。
        将差值矩阵带入高斯函数的公式中，计算每个坐标点的高斯值。
        最终得到的 heatmap 即为生成的二维高斯热图，表示了关键点在图像上的分布情况。
        :param joint_coord_img:  像素坐标位置
        :return: 生成的二维高斯热图，表示了关键点在图像上的分布情况
        """
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        yy, xx = torch.meshgrid(y, x)
        xx = xx[None, None, :, :].cuda().float()
        yy = yy[None, None, :, :].cuda().float()

        x = joint_coord_img[:, :, 0, None, None]
        y = joint_coord_img[:, :, 1, None, None]
        heatmap = torch.exp(
            -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2)
        return heatmap

    def get_coord(self, smpl_pose, smpl_shape, smpl_trans):
        batch_size = smpl_pose.shape[0]
        mesh_cam, _ = self.human_model_layer(smpl_pose, smpl_shape, smpl_trans)
        # camera-centered 3D coordinate
        joint_cam = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None, :, :].repeat(batch_size, 1, 1),
                              mesh_cam)
        root_joint_idx = self.human_model.root_joint_idx

        # project 3D coordinates to 2D space
        x = joint_cam[:, :, 0] / (joint_cam[:, :, 2] + 1e-4) * cfg.focal[0] + cfg.princpt[0]
        y = joint_cam[:, :, 1] / (joint_cam[:, :, 2] + 1e-4) * cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        mesh_cam_render = mesh_cam.clone()
        # root-relative 3D coordinates
        root_cam = joint_cam[:, root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam - root_cam
        return joint_proj, joint_cam, mesh_cam, mesh_cam_render

    def forward(self, inputs, targets, meta_info, mode):
        early_img_feat = self.backbone(inputs['img'])  # pose_guided_img_feat
        """
        模型解释： 
            1. 将bs x 3 x 256 x 256的原始图片输入早期的特征提取网络中，得到 Bs x 64 x 64 x 64 得到图片早期特征。
            2. 得到数据集中的真实 2D 关节像素坐标。 这些坐标是在数据处理时将真实关键点缩放到 64 x 64 图片比例中，还要检查截断。
            3. 将2D关节像素坐标生成二维高斯热图，表示了关键点在图像上的分布情况。
            4. 对这些关节热图进行 关节掩码检测 是否被遮挡或者没有  Js = 30 表示由多个数据集定义的关节集的超集中的关节数。
                我们通过将零乘以相应的关节的热图，将不关心值分配给未定义的关节和推理时间置信度较低的关节预测。
                基于关节和热图的超集建模不关心值，使3DCrowdNet能够使用单个网络对各种人体关节集进行推理，
                并处理不同的输入，如由于截断和遮挡导致关节缺失的二维姿态
            5. 将这个二维特征 与 早期图片特征进行拼接通过  pose2feat网络  得到 Bs x 64 x 64 x 64
            6. 将上述特征送入剩下的 resnet-50 的剩余网络进行特征提取 得到 Bs x 2048 x 8 x 8
            7. 将这个特送入 PositionNet 中得到 从图像中预测出的关键点热图中估计关键点的三维位置和得分。
                实现： (1)通过一个 1x1 的卷积层 调整 通道数 为 15 x 8 再 reshape bs x 15 x 8 x 8 x 8
                      (2)通过 soft_argmax_3d 操作 得到 关键点的三维坐标
                      (3)将关键点的坐标归一化到 [-1, 1] 范围内
                      (4)使用 F.grid_sample 函数在关键点热图上进行采样，得到每个关键点的得分
                三维关键点坐标 (batch_size, 15, 3) 
                关键点得分  (batch_size, 15, 1)   每个关节点位置 在特征图上的像素特征值
            8.  将pose_guided_img_feat (batch_size x 2048 x 8 x 8)，joint_img (batch_size, 15, 3)，joint_score
                (batch_size, 15, 1)送入rotation_net。
                实现： (1) 根据 pose_guided_img_feat 以及 三维关键点坐标 的 x，y 坐标在特征图像上进行采样 得到 
                        (batch_size, joint_num, channel_dim) 大小的特征  相当于 每个关节点位置 在特征图上的像素特征值。
                      (2) 再将 三维关键点坐标，三维关键点位置在三维特征图上的像素特征值，二维关键点位置在二维特征图上的像素特征值在
                         在 第二维度 拼接起来 (batch_size,15, 2048 + 3 + 1)
                      (3) 再将这些特征 送入 构建的图卷积网络 
                          图卷积网络块构成：
                            两层 GraphConvBlock 组成 残差连接  共叠加 4 次 
                            GraphConvBlock 构成:
                                为每一个关节点 构建 全连接层
                      (4) 特征通过 图卷积网络 得到 (batch_size, 15 , 128)
                      (5）对特征重塑为 (batch_size, 15 * 128) 再通过各自回归参数的全连接层。
            9.得到各种需要的参数          
        """
        # get pose guided image feature  ---- 编码器部分
        joint_coord_img = inputs['joints']
        with torch.no_grad():
            joint_heatmap = self.make_2d_gaussian_heatmap(joint_coord_img.detach())
            # remove blob centered at (0,0) == invalid ones
            joint_heatmap = joint_heatmap * inputs['joints_mask'][:, :, :, None]
        pose_img_feat = self.pose2feat(early_img_feat, joint_heatmap)
        pose_guided_img_feat = self.backbone(pose_img_feat, skip_early=True)  # 2048 x 8 x 8
        pose_guided_img_feat = self.module(pose_guided_img_feat)  # 2048 x 8 x 8

        # 解码器部分
        joint_img, joint_score = self.position_net(pose_guided_img_feat)  # refined 2D pose or 3D pose
        # estimate model parameters
        root_pose_6d, z, shape_param, cam_param = self.rotation_net(pose_guided_img_feat, joint_img.detach(),
                                                                    joint_score.detach())
        # change root pose 6d + latent code -> axis angles
        root_pose = rot6d_to_axis_angle(root_pose_6d)  # 6D 旋转表示 转换为角轴的形式
        pose_param = self.vposer(z)
        cam_trans = self.get_camera_trans(cam_param, meta_info, is_render=(cfg.render and (mode == 'test')))
        pose_param = pose_param.view(-1, self.human_model.orig_joint_num - 1, 3)
        pose_param = torch.cat((root_pose[:, None, :], pose_param), 1).view(-1, self.human_model.orig_joint_num * 3)
        joint_proj, joint_cam, mesh_cam, mesh_cam_render = self.get_coord(pose_param, shape_param, cam_trans)

        if mode == 'train':
            # loss functions
            loss = {}
            # joint_img: 0~8, joint_proj: 0~64, target: 0~64
            loss['body_joint_img'] = (1 / 8) * self.coord_loss(joint_img * 8, self.human_model.reduce_joint_set(
                targets['orig_joint_img']), self.human_model.reduce_joint_set(meta_info['orig_joint_trunc']),
                                                               meta_info['is_3D'])
            loss['smpl_joint_img'] = (1 / 8) * self.coord_loss(joint_img * 8, self.human_model.reduce_joint_set(
                targets['fit_joint_img']),
                                                               self.human_model.reduce_joint_set(
                                                                   meta_info['fit_joint_trunc']) * meta_info[
                                                                                                       'is_valid_fit'][
                                                                                                   :, None, None])
            loss['smpl_pose'] = self.param_loss(pose_param, targets['pose_param'],
                                                meta_info['fit_param_valid'] * meta_info['is_valid_fit'][:, None])
            loss['smpl_shape'] = self.param_loss(shape_param, targets['shape_param'],
                                                 meta_info['is_valid_fit'][:, None])
            loss['body_joint_proj'] = (1 / 8) * self.coord_loss(joint_proj, targets['orig_joint_img'][:, :, :2],
                                                                meta_info['orig_joint_trunc'])
            loss['body_joint_cam'] = self.coord_loss(joint_cam, targets['orig_joint_cam'],
                                                     meta_info['orig_joint_valid'] * meta_info['is_3D'][:, None, None])
            loss['smpl_joint_cam'] = self.coord_loss(joint_cam, targets['fit_joint_cam'],
                                                     meta_info['is_valid_fit'][:, None, None])

            return loss

        else:
            # test output
            out = {'cam_param': cam_param}
            # out['input_joints'] = joint_coord_img
            out['joint_img'] = joint_img * 8
            out['joint_proj'] = joint_proj
            out['joint_score'] = joint_score
            out['smpl_mesh_cam'] = mesh_cam
            out['smpl_pose'] = pose_param
            out['smpl_shape'] = shape_param

            out['mesh_cam_render'] = mesh_cam_render

            if 'smpl_mesh_cam' in targets:
                out['smpl_mesh_cam_target'] = targets['smpl_mesh_cam']
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            if 'img2bb_trans' in meta_info:
                out['img2bb_trans'] = meta_info['img2bb_trans']
            if 'bbox' in meta_info:
                out['bbox'] = meta_info['bbox']
            if 'tight_bbox' in meta_info:
                out['tight_bbox'] = meta_info['tight_bbox']
            if 'aid' in meta_info:
                out['aid'] = meta_info['aid']

            return out


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


def get_model(vertex_num, joint_num, mode):
    backbone = ResNetBackbone(cfg.resnet_type)
    module = MidAttention()
    pose2feat = Pose2Feat(joint_num)
    position_net = PositionNet()
    rotation_net = RotationNet()
    vposer = Vposer()

    if mode == 'train':
        backbone.init_weights()
        # backbone.apply(init_weights)
        module.init_weights()
        pose2feat.apply(init_weights)
        position_net.apply(init_weights)
        rotation_net.apply(init_weights)

    model = Model(backbone, pose2feat, position_net, rotation_net, vposer, module)  # module
    return model


def get_mesh_model(vertex_num, joint_num, mode):
    backbone = ResNetBackbone(cfg.resnet_type)
    module = MidAttention()
    pose2feat = Pose2Feat(joint_num)
    position_net = PositionNet()
    rotation_net = RotationNet()
    vposer = Vposer()

    if mode == 'train':
        backbone.init_weights()
        # backbone.apply(init_weights)
        module.init_weights()
        pose2feat.apply(init_weights)
        position_net.apply(init_weights)
        rotation_net.apply(init_weights)

    model = Model(backbone, pose2feat, position_net, rotation_net, vposer, module)  # module
    return model
