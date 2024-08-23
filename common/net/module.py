import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import cfg
from human_body_prior.tools.model_loader import load_vposer
import torchgeometry as tgm
from common.net.layer import make_conv_layers, make_deconv_layers, make_conv1d_layers, make_linear_layers, \
    GraphConvBlock, \
    GraphResBlock, Conv, DilatedConv
from common.utils.mano import MANO
from common.utils.smpl import SMPL
import numpy as np
from einops import rearrange, repeat
from common.net.transformer import build_transformer_encoder, PositionEmbeddingSine, build_transformer, \
    TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from typing import Optional, List
from torch import nn, Tensor


class Pose2Feat(nn.Module):
    def __init__(self, joint_num):
        super(Pose2Feat, self).__init__()
        self.joint_num = joint_num
        self.conv = make_conv_layers([64 + joint_num, 64])

    def forward(self, img_feat, joint_heatmap):
        feat = torch.cat((img_feat, joint_heatmap), 1)
        feat = self.conv(feat)
        return feat


class PositionNet(nn.Module):
    def __init__(self):
        super(PositionNet, self).__init__()
        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.human_model = MANO()
            self.joint_num = self.human_model.graph_joint_num
        else:
            self.human_model = SMPL()
            self.joint_num = self.human_model.graph_joint_num  # 15 表示由多个数据集定义的关节集交集中的关节数

        self.hm_shape = [cfg.output_hm_shape[0] // 8, cfg.output_hm_shape[1] // 8, cfg.output_hm_shape[2] // 8]
        self.conv = make_conv_layers([2048, self.joint_num * self.hm_shape[0]], kernel=1, stride=1, padding=0,
                                     bnrelu_final=False)

    def soft_argmax_3d(self, heatmap3d):
        """
        目的是根据热图中每个关节的概率分布来计算其在三维空间中的位置坐标
        这个函数实现了一个三维空间中的软最大化函数（soft argmax）。
        它的输入是一个三维热图（heatmap3d），其形状为 (batch_size, joint_num, height, width, depth)，其中：
            batch_size 表示批次大小；
            joint_num 表示关节数量；
            height、width 和 depth 分别表示热图在三个维度上的大小。
        该函数的操作步骤如下：
            将输入的三维热图重塑为 (batch_size * joint_num, height * width * depth) 的形状，并对最后一个维度进行 softmax 操作，以获得归一化的概率分布。
            将概率分布重塑回原来的形状 (batch_size, joint_num, height, width, depth)。
            分别在 x、y 和 z 方向上对概率分布进行求和，得到三个向量 accu_x、accu_y 和 accu_z。
            将 accu_x、accu_y 和 accu_z 分别与对应的维度坐标相乘，再在 x、y 和 z 方向上进行求和，得到坐标输出 coord_out。
        最后将 x、y 和 z 方向上的坐标合并，返回最终的坐标输出。
        :param heatmap3d: 3D 热图
        :return: 最终三维坐标输出
        """
        heatmap3d = heatmap3d.reshape((-1, self.joint_num, self.hm_shape[0] * self.hm_shape[1] * self.hm_shape[2]))
        heatmap3d = F.softmax(heatmap3d, 2)
        heatmap3d = heatmap3d.reshape((-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2]))

        accu_x = heatmap3d.sum(dim=(2, 3))
        accu_y = heatmap3d.sum(dim=(2, 4))
        accu_z = heatmap3d.sum(dim=(3, 4))

        accu_x = accu_x * torch.arange(self.hm_shape[2]).float().cuda()[None, None, :]
        accu_y = accu_y * torch.arange(self.hm_shape[1]).float().cuda()[None, None, :]
        accu_z = accu_z * torch.arange(self.hm_shape[0]).float().cuda()[None, None, :]

        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)
        accu_z = accu_z.sum(dim=2, keepdim=True)

        coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
        return coord_out

    def forward(self, img_feat):
        """
        实现了一种通过预测的关键点热图来估计关键点的三维坐标和得分的方法。具体操作如下：
        通过卷积操作 self.conv(img_feat) 将图像特征转换成关键点热图，形状为 (batch_size, joint_num, hm_shape[0], hm_shape[1], hm_shape[2])。
        调用 soft_argmax_3d 方法计算关键点的三维坐标，得到 joint_coord，形状为 (batch_size, joint_num, 3)。
        对于每个关键点，计算其在关键点热图上的采样得分。首先将关键点的坐标归一化到 [-1, 1] 范围内，
        然后使用 F.grid_sample 函数在关键点热图上进行采样，得到每个关键点的得分。将所有关键点的得分存储在 scores 中。
        将得分堆叠成形状为 (joint_num, batch_size) 的张量，并进行转置，得到形状为 (batch_size, joint_num, 1) 的 joint_score。
        返回关键点的三维坐标 joint_coord 和对应的得分 joint_score。
        这种方法可以用于从图像中预测出的关键点热图中估计关键点的三维位置和得分。
        """
        # joint heatmap
        joint_heatmap = self.conv(img_feat).view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1],
                                                 self.hm_shape[2])
        # joint_heatmap = self.conv(img_feat).reshape(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1],
        #                                          self.hm_shape[2])

        # joint coord
        joint_coord = self.soft_argmax_3d(joint_heatmap)  # bs x 15 x 3

        # joint score sampling
        scores = []
        joint_heatmap = joint_heatmap.view(-1, self.joint_num, self.hm_shape[0] * self.hm_shape[1] * self.hm_shape[2])
        # joint_heatmap = joint_heatmap.reshape(-1, self.joint_num, self.hm_shape[0] * self.hm_shape[1] * self.hm_shape[2])
        joint_heatmap = F.softmax(joint_heatmap, 2)
        joint_heatmap = joint_heatmap.view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        # joint_heatmap = joint_heatmap.reshape(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        for j in range(self.joint_num):
            x = joint_coord[:, j, 0] / (self.hm_shape[2] - 1) * 2 - 1
            y = joint_coord[:, j, 1] / (self.hm_shape[1] - 1) * 2 - 1
            z = joint_coord[:, j, 2] / (self.hm_shape[0] - 1) * 2 - 1
            grid = torch.stack((x, y, z), 1)[:, None, None, None, :]
            score_j = F.grid_sample(joint_heatmap[:, j, None, :, :, :], grid, align_corners=True)[:, 0, 0, 0,
                      0]  # (batch_size)
            scores.append(score_j)
        scores = torch.stack(scores)  # (joint_num, batch_size)
        joint_score = scores.permute(1, 0)[:, :, None]  # (batch_size, joint_num, 1)
        return joint_coord, joint_score  # (bs x joint_num x 3),(bs x joint_num,1)


class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()

        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.human_model = MANO()
            self.joint_num = self.human_model.graph_joint_num
            self.graph_adj = torch.from_numpy(self.human_model.graph_adj).float()
        else:
            self.human_model = SMPL()
            self.joint_num = self.human_model.graph_joint_num  # 15
            self.graph_adj = torch.from_numpy(self.human_model.graph_adj).float()  # 构建的邻接矩阵

        # graph convs
        self.graph_block = nn.Sequential(*[ \
            GraphConvBlock(self.graph_adj, 2048 + 4, 128),
            GraphResBlock(self.graph_adj, 128),
            GraphResBlock(self.graph_adj, 128),
            GraphResBlock(self.graph_adj, 128),
            GraphResBlock(self.graph_adj, 128)])

        # self.graph_block_1 = nn.Sequential(
        #     *[GraphConvBlock(self.graph_adj, 2048, 128),
        #       GraphResBlock(self.graph_adj, 128),
        #       GraphResBlock(self.graph_adj, 128),
        #       GraphResBlock(self.graph_adj, 128),
        #       GraphResBlock(self.graph_adj, 128)])

        # self.conv = GraphConvBlock(self.graph_adj, 256, 128)
        self.hm_shape = [cfg.output_hm_shape[0] // 8, cfg.output_hm_shape[1] // 8, cfg.output_hm_shape[2] // 8]

        self.root_pose_out = make_linear_layers([self.joint_num * 128, 6], relu_final=False)
        self.pose_out = make_linear_layers([self.joint_num * 128, self.human_model.v_poser_code_dim],
                                           relu_final=False)  # Vposer latent code
        self.shape_out = make_linear_layers([self.joint_num * 128, self.human_model.shape_param_dim], relu_final=False)
        self.cam_out = make_linear_layers([self.joint_num * 128, 3], relu_final=False)

    def sample_image_feature(self, img_feat, joint_coord_img):
        """
        这个函数接受图像特征(img_feat)、图像空间中的关节坐标(joint_coord_img)作为输入。
        它从图像(img_feat)中根据图像中的关节坐标进行特征采样。
        它将关节坐标转换为网格坐标，使用F.grid_sample进行特征采样，并堆叠采样特征。
        输出是与关节对应的图像特征的张 量。
        :param img_feat: 图像特征
        :param joint_coord_img: 图像空间中的关节坐标
        :return: 与关节对应的图像特征的张量
        """
        img_feat_joints = []
        for j in range(self.joint_num):
            x = joint_coord_img[:, j, 0] / (self.hm_shape[2] - 1) * 2 - 1
            y = joint_coord_img[:, j, 1] / (self.hm_shape[1] - 1) * 2 - 1
            grid = torch.stack((x, y), 1)[:, None, None, :]
            img_feat = img_feat.float()
            img_feat_j = F.grid_sample(img_feat, grid, align_corners=True)[:, :, 0, 0]  # (batch_size, channel_dim)
            img_feat_joints.append(img_feat_j)
        img_feat_joints = torch.stack(img_feat_joints)  # (joint_num, batch_size, channel_dim)
        img_feat_joints = img_feat_joints.permute(1, 0, 2)  # (batch_size, joint_num, channel_dim)
        return img_feat_joints

    def forward(self, img_feat, joint_coord_img, joint_score):
        # pose parameter
        img_feat_joints = self.sample_image_feature(img_feat, joint_coord_img)  # (bs x joint_num x 2048)
        # pose_2d_feat = self.graph_block_1(img_feat_joints)
        feat = torch.cat((img_feat_joints, joint_coord_img, joint_score), 2)  # bs x joint_num x channel
        feat = self.graph_block(feat)  # bs x joint_num x 128
        # feat = torch.cat((pose_2d_feat, feat), dim=2)
        # feat = self.conv(feat)
        root_pose = self.root_pose_out(feat.view(-1, self.joint_num * 128))  # bs x 6
        pose_param = self.pose_out(feat.view(-1, self.joint_num * 128))  # bs x 32
        # shape parameter
        shape_param = self.shape_out(feat.view(-1, self.joint_num * 128))  # bs x 10
        # camera parameter
        cam_param = self.cam_out(feat.view(-1, self.joint_num * 128))  # bs x 3

        # return root_pose, pose_param, shape_param, cam_param
        return root_pose, pose_param, shape_param, cam_param


class Vposer(nn.Module):
    """
    这段代码定义了一个名为 Vposer 的神经网络模型，该模型用于处理姿势参数 z 并输出对应的身体姿势。
    在 __init__ 方法中
        模型初始化时加载了预训练的 VPOSER 模型。通过调用 load_vposer 函数来加载 VPOSER 模型，该函数返回一个模型对象和其他辅助信息。
        然后将模型设为评估模式（eval()）。
    在 forward 方法中，接受一个表示姿势参数 z 的输入。首先获取输入张 量的批量大小，
        然后使用 VPOSER 模型的 decode 方法将姿势参数 z 解码成关节角度表示的身体姿势。
        output_type='aa' 表示输出为欧拉角度。
        将结果视图调整为 (batch_size, 24 - 3, 3) 的形状，其中 24 - 3 表示除了根部、右手和左手之外的关节数量。
        接着创建一个全零张量 zero_pose 表示零手部姿势，并将其连接到身体姿势中。最后将结果重塑为 (batch_size, -1) 的形状并返回。
    这个模型接受一个姿势参数 z 作为输入，通过预训练的 VPOSER 模型解码并输出对应的身体姿势。
    """

    def __init__(self):
        super(Vposer, self).__init__()
        self.vposer, _ = load_vposer(osp.join(cfg.human_model_path, 'smpl', 'VPOSER_CKPT'), vp_model='snapshot')
        self.vposer.eval()

    def forward(self, z):
        batch_size = z.shape[0]
        body_pose = self.vposer.decode(z, output_type='aa').view(batch_size, -1).view(-1, 24 - 3,
                                                                                      3)  # without root, R_Hand, L_Hand
        zero_pose = torch.zeros((batch_size, 1, 3)).float().cuda()

        # attach zero hand poses
        body_pose = torch.cat((body_pose, zero_pose, zero_pose), 1)
        body_pose = body_pose.view(batch_size, -1)
        return body_pose


class PositionalEncoding1D(nn.Module):

    def __init__(self, d_hid=256, n_position=512):
        super(PositionalEncoding1D, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    @staticmethod
    def _get_sinusoid_encoding_table(n_position, d_hid):
        """Sinusoid position encoding table"""

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table)  # N, C

    def forward(self, x):
        pass


class Liftingnet(nn.Module):
    def __init__(self):
        super(Liftingnet, self).__init__()
        self.exponent = nn.Parameter(torch.tensor(3, dtype=torch.float32))
        self.conv2d_to_3d = nn.Conv2d(2048, 256 * 8, 1, 1)
        self.conv_3d_coord = nn.Sequential(nn.Conv3d(256 + 3, 256, 1, 1))
        self.pos_embed_1d = PositionalEncoding1D()
        self.refine_3d = build_transformer_encoder(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            num_encoder_layers=cfg.enc_layers,
            normalize_before=False,
        )

    @staticmethod
    def get_relative_depth_anchour(k, map_size=8):
        """
        创建一个范围为 [0, 1) 的一维张量 rangearr，表示地图上的归一化位置。
        通过 rangearr 创建 Y_map 和 X_map 张 量，分别表示地图上的 Y 坐标和 X 坐标。
        使用参数 k 对 rangearr 进行指数运算，得到 Z_map 张 量，表示深度坐标。
        将 Z_map、Y_map 和 X_map 沿着通道维度拼接起来，得到形状为 (1, 3, 8, 8, 8) 的相对深度锚点张 量。

        函数的返回结果将在某些深度感知任务中用作参考点,用于定义物体或场景的相对深度。
        """
        rangearr = torch.arange(map_size, dtype=torch.float32, device=k.device) / map_size  # (0, 1)
        Y_map = rangearr.reshape(1, 1, 1, map_size, 1).repeat(1, 1, map_size, 1, map_size)
        X_map = rangearr.reshape(1, 1, 1, 1, map_size).repeat(1, 1, map_size, map_size, 1)
        Z_map = torch.pow(rangearr, k)
        Z_map = Z_map.reshape(1, 1, map_size, 1, 1).repeat(1, 1, 1, map_size, map_size)
        return torch.cat([Z_map, Y_map, X_map], dim=1)  # 1, 3, 8, 8, 8

    def forward(self, img_feature, inputs_img):
        dense_feat = self.conv2d_to_3d(img_feature)
        dense_feat = rearrange(dense_feat, 'b (c d) h w -> b c d h w', c=256, d=8)
        exponent = torch.clamp(self.exponent, 1, 20)
        relative_depth_anchour = self.get_relative_depth_anchour(exponent)
        cam_anchour_maps = repeat(relative_depth_anchour, 'n c d h w -> (b n) c d h w', b=dense_feat.size(0))
        dense_feat = torch.cat([dense_feat, cam_anchour_maps], dim=1)
        dense_feat = self.conv_3d_coord(dense_feat)
        dense_feat = rearrange(dense_feat, 'b c d h w -> (d h w) b c', c=256, d=8).contiguous()
        pos_3d = repeat(self.pos_embed_1d.pos_table, 'n c -> n b c', b=inputs_img.size(0))
        dense_feat = self.refine_3d(dense_feat, pos=pos_3d)
        dense_feat = rearrange(dense_feat, '(d h w) b c -> b c d h w', d=8, h=8, w=8).contiguous()
        return dense_feat, exponent


class Hide2dfeature(nn.Module):
    def __init__(self):
        super(Hide2dfeature, self).__init__()
        self.down_linear = nn.Conv2d(2048, 256, 1, 1)
        self.pos_embed = PositionEmbeddingSine(128, normalize=True)

    def forward(self, img_feature):
        img = self.down_linear(img_feature)
        with torch.no_grad():
            img_pos = self.pos_embed(img)
            src_masks = None
        img = rearrange(img, 'b c h w -> (h w) b c')
        img_pos = rearrange(img_pos, 'b c h w -> (h w) b c')
        return img, img_pos, src_masks


class Fusiontransformer(nn.Module):
    def __init__(self):
        super(Fusiontransformer, self).__init__()
        self.query = nn.Embedding(15, 256)
        self.spose_shape_cam_param = nn.Parameter(torch.randn(3, 256).float())
        self.transformer = build_transformer(
            d_model=256,
            nhead=8,
            num_encoder_layers=cfg.enc_layers,
            num_decoder_layers=cfg.dec_layers,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=True
        )

    def forward(self, img, src_masks, img_pos, dense_feat, exponent, inputs_img):
        query_pos = self.query.weight
        query_pos = torch.cat([self.spose_shape_cam_param, query_pos], dim=0)
        query_pos = repeat(query_pos, 'n c -> n b c', b=inputs_img.shape[0])
        hs, joint_img = self.transformer(img, src_masks, query_pos, img_pos, dense_feat, exponent)
        hs = rearrange(hs, 'l n b c -> l b n c')
        pose_token, shape_token, cam_token = hs[:, :, 0], hs[:, :, 1], hs[:, :, 2]
        return pose_token, shape_token, cam_token, joint_img


class Cascadenet(nn.Module):
    def __init__(self, num_layers):
        super(Cascadenet, self).__init__()
        self.cascade_root_pose_out = nn.ModuleList(
            [nn.Linear(256, 6)] + \
            [nn.Linear(256 + 6, 6) for _ in range(num_layers - 1)]
        )
        self.cascade_pose_out = nn.ModuleList(
            [nn.Linear(256, 32)] + \
            [nn.Linear(256 + 32, 32) for _ in range(num_layers - 1)]
        )
        self.cascade_shaoe_out = nn.ModuleList(
            [nn.Linear(256, 10)] + \
            [nn.Linear(256 + 10, 10) for _ in range(num_layers - 1)]
        )
        self.cascade_cam_out = nn.ModuleList(
            [nn.Linear(256, 3)] + \
            [nn.Linear(256 + 3, 3) for _ in range(num_layers - 1)]
        )

    @staticmethod
    def cascade_fc(hs, net_list):
        assert len(hs) == len(net_list)
        for i in range(len(hs)):
            if i == 0:
                out = net_list[i](hs[i])
            else:
                offset = net_list[i](torch.cat([hs[i], out], dim=-1))
                out = out + offset
        return out

    def forward(self, pose_token, shape_token, cam_token):
        root_pose_6d = self.cascade_fc(pose_token, self.cascade_root_pose_out)  # bs x 6
        pose_param = self.cascade_fc(pose_token, self.cascade_pose_out)  # bs x 32
        shape_param = self.cascade_fc(shape_token, self.cascade_shaoe_out)  # bs x 10
        cam_param = self.cascade_fc(cam_token, self.cascade_cam_out)  # bs x 3
        return root_pose_6d, pose_param, shape_param, cam_param


class MutilScaleModule(nn.Module):
    def __init__(self):
        super(MutilScaleModule, self).__init__()
        self.dims = [128, 256, 512]
        self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 2, 4]]

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(Conv(64, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True)))
        for i in range(2):
            downsample_layer = nn.Sequential(
                Conv(self.dims[i] * 2, self.dims[i + 1], kSize=3, stride=2, padding=1, bn_act=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(3):
            stage_blocks = []
            for j in range(3):
                if i < 2:
                    stage_blocks.append(DilatedConv(dim=self.dims[i], k=3, dilation=self.dilation[i][j], drop_path=0.2))
                else:
                    stage_blocks.append(DilatedConv(dim=self.dims[i], k=3, dilation=self.dilation[i][j], drop_path=0.2))
                    if j == 2:
                        stage_blocks.append(
                            DilatedConv(dim=self.dims[i], k=3, dilation=self.dilation[i][j + 1], drop_path=0.2))
            self.stages.append(nn.Sequential(*stage_blocks))

    def forward(self, x):
        x_down = []
        x = self.downsample_layers[0](x)  # 128 x 32 x 32
        x_down.append(x)

        feature_x = []
        for i in range(3):
            x = self.stages[0][i](x)  # 128 x 32 x 32
        feature_x.append(x)

        x = torch.cat((x_down[0], feature_x[0]), dim=1)
        x = self.downsample_layers[1](x)  # 256 x 16 x 16
        x_down.append(x)
        for i in range(3):
            x = self.stages[1][i](x)  # 256 x 16 x 16
        feature_x.append(x)

        x = torch.cat((x_down[1], feature_x[1]), dim=1)
        x = self.downsample_layers[2](x)  # 512 x 8 x 8
        x_down.append(x)
        for i in range(3):
            x = self.stages[2][i](x)  # 512 x 8 x 8
        feature_x.append(x)

        return feature_x


class CrossFeature(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(CrossFeature, self).__init__()
        self.cnv = nn.Conv2d(input_channel, out_channel, stride=1, kernel_size=1)

    def forward(self, x):
        x = self.cnv(x)
        return x


class MultiscaleTransformer(nn.Module):
    def __init__(self, num_encoder_layers=1,
                 dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False):
        super(MultiscaleTransformer, self).__init__()
        self.dim = [128, 256, 512]
        self.down_linear_1 = nn.Conv2d(64, self.dim[0], 3, 2, padding=1)
        self.pos_embed_1 = PositionEmbeddingSine(64, normalize=True)

        encoder_layer_1 = TransformerEncoderLayer(128, 8, dim_feedforward,
                                                  dropout, activation, normalize_before)
        encoder_norm_1 = nn.LayerNorm(128) if normalize_before else None
        self.encoder_1 = TransformerEncoder(encoder_layer_1, num_encoder_layers, encoder_norm_1)

        self.cross_feature = nn.ModuleList()
        for i in range(3):
            self.cross_feature.append(nn.Sequential(CrossFeature(self.dim[i] * 2, self.dim[i])))

        self.down_linear_2 = nn.Conv2d(self.dim[0], self.dim[1], 3, 2, padding=1)
        self.pos_embed_2 = PositionEmbeddingSine(128, normalize=True)

        encoder_layer_2 = TransformerEncoderLayer(256, 8, dim_feedforward,
                                                  dropout, activation, normalize_before)
        encoder_norm_2 = nn.LayerNorm(256) if normalize_before else None
        self.encoder_2 = TransformerEncoder(encoder_layer_2, num_encoder_layers, encoder_norm_2)

        self.down_linear_3 = nn.Conv2d(self.dim[1], self.dim[2], 3, 2, padding=1)
        self.pos_embed_3 = PositionEmbeddingSine(256, normalize=True)

        encoder_layer_3 = TransformerEncoderLayer(512, 8, dim_feedforward,
                                                  dropout, activation, normalize_before)
        encoder_norm_3 = nn.LayerNorm(512) if normalize_before else None
        self.encoder_3 = TransformerEncoder(encoder_layer_3, num_encoder_layers, encoder_norm_3)
        self.conv = nn.Conv2d(512, 256, kernel_size=1, stride=1)

        self.exponent = nn.Parameter(torch.tensor(3, dtype=torch.float32))
        self.conv2d_to_3d = nn.Conv2d(512, 64 * 8, 1, 1)
        self.conv_3d_coord = nn.Sequential(nn.Conv3d(64 + 3, 64, 1, 1),
                                           nn.Conv3d(64, 256, 1, 1))
        self.pos_embed_1d = PositionalEncoding1D()
        self.refine_3d = build_transformer_encoder(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            num_encoder_layers=1,
            normalize_before=False,
        )

    @staticmethod
    def get_relative_depth_anchour(k, map_size=8):
        rangearr = torch.arange(map_size, dtype=torch.float32, device=k.device) / map_size  # (0, 1)
        Y_map = rangearr.reshape(1, 1, 1, map_size, 1).repeat(1, 1, map_size, 1, map_size)
        X_map = rangearr.reshape(1, 1, 1, 1, map_size).repeat(1, 1, map_size, map_size, 1)
        Z_map = torch.pow(rangearr, k)
        Z_map = Z_map.reshape(1, 1, map_size, 1, 1).repeat(1, 1, 1, map_size, map_size)
        return torch.cat([Z_map, Y_map, X_map], dim=1)  # 1, 3, 8, 8, 8

    def forward(self, x, multiscale_feature):
        img_1 = self.down_linear_1(x)
        with torch.no_grad():
            img_pos_1 = self.pos_embed_1(img_1)
            src_masks = None
        img_1 = rearrange(img_1, 'b c h w -> (h w) b c')
        img_pos_1 = rearrange(img_pos_1, 'b c h w -> (h w) b c')
        memory_1 = self.encoder_1(img_1, src_key_padding_mask=src_masks, pos=img_pos_1)
        h_1, w_1 = 32, 32
        memory_1 = rearrange(memory_1, '(h w) b c -> b c h w', h=h_1, w=w_1)
        feature_1 = torch.cat((memory_1, multiscale_feature[0]), dim=1)
        feature_1 = self.cross_feature[0](feature_1)

        img_2 = self.down_linear_2(feature_1)
        with torch.no_grad():
            img_pos_2 = self.pos_embed_2(img_2)
            src_masks = None
        img_2 = rearrange(img_2, 'b c h w -> (h w) b c')
        img_pos_2 = rearrange(img_pos_2, 'b c h w -> (h w) b c')
        memory_2 = self.encoder_2(img_2, src_key_padding_mask=src_masks, pos=img_pos_2)
        h_2, w_2 = 16, 16
        memory_2 = rearrange(memory_2, '(h w) b c -> b c h w', h=h_2, w=w_2)
        feature_2 = torch.cat((memory_2, multiscale_feature[1]), dim=1)
        feature_2 = self.cross_feature[1](feature_2)

        img_3 = self.down_linear_3(feature_2)
        with torch.no_grad():
            img_pos_3 = self.pos_embed_3(img_3)
            src_masks = None
        img_3 = rearrange(img_3, 'b c h w -> (h w) b c')
        img_pos_3 = rearrange(img_pos_3, 'b c h w -> (h w) b c')
        memory_3 = self.encoder_3(img_3, src_key_padding_mask=src_masks, pos=img_pos_3)
        h_3, w_3 = 8, 8
        memory_3 = rearrange(memory_3, '(h w) b c -> b c h w', h=h_3, w=w_3)
        feature_3 = torch.cat((memory_3, multiscale_feature[2]), dim=1)
        feature_3 = self.cross_feature[2](feature_3)

        dense_feat = self.conv2d_to_3d(feature_3)
        dense_feat = rearrange(dense_feat, 'b (c d) h w -> b c d h w', c=64, d=8)
        exponent = torch.clamp(self.exponent, 1, 20)
        relative_depth_anchour = self.get_relative_depth_anchour(exponent)
        cam_anchour_maps = repeat(relative_depth_anchour, 'n c d h w -> (b n) c d h w', b=dense_feat.size(0))
        dense_feat = torch.cat([dense_feat, cam_anchour_maps], dim=1)
        dense_feat = self.conv_3d_coord(dense_feat)
        dense_feat = rearrange(dense_feat, 'b c d h w -> (d h w) b c', c=256, d=8).contiguous()
        pos_3d = repeat(self.pos_embed_1d.pos_table, 'n c -> n b c', b=x.size(0))
        dense_feat = self.refine_3d(dense_feat, pos=pos_3d)
        dense_feat = rearrange(dense_feat, '(d h w) b c -> b c d h w', d=8, h=8, w=8).contiguous()

        feature_3 = self.conv(feature_3)
        with torch.no_grad():
            img_pos = self.pos_embed_2(feature_3)
        img_pos = rearrange(img_pos, 'b c h w -> (h w) b c')

        return img_pos, feature_3, dense_feat, exponent


class TransDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_decoder_layers=3, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self.smpl_tgt = nn.Parameter(torch.zeros(256))
        self.kp_tgt = nn.Parameter(torch.zeros(256))

    def forward(self, src, query_embed, pos_embed, dense_feat, exponent, mask=None):
        L, B, C = query_embed.shape
        smpl_tgt = repeat(self.smpl_tgt, 'c -> n b c', n=3, b=B)
        kp_tgt = repeat(self.kp_tgt, 'c ->n b c', n=15, b=B)
        tgt = torch.cat([smpl_tgt, kp_tgt], dim=0)
        memory = rearrange(src, 'b c h w -> (h w) b c')
        hs, joint_img = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                     pos=pos_embed, query_pos=query_embed, dense_feat=dense_feat, exponent=exponent)
        return hs, joint_img


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.query = nn.Embedding(15, 256)
        self.spose_shape_cam_param = nn.Parameter(torch.randn(3, 256).float())
        self.transformer = TransDecoder()

    def forward(self, img, img_pos, dense_feat, exponent, inputs_img):
        query_pos = self.query.weight
        query_pos = torch.cat([self.spose_shape_cam_param, query_pos], dim=0)
        query_pos = repeat(query_pos, 'n c -> n b c', b=inputs_img.shape[0])
        hs, joint_img = self.transformer(img, query_pos, img_pos, dense_feat, exponent)
        hs = rearrange(hs, 'l n b c -> l b n c')
        pose_token, shape_token, cam_token = hs[:, :, 0], hs[:, :, 1], hs[:, :, 2]
        return pose_token, shape_token, cam_token, joint_img
