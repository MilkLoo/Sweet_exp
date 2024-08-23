import torch
import numpy as np
from config import cfg
import torchgeometry as tgm
from torch.nn import functional as F


def build_adj(vertex_num, skeleton, flip_pairs):
    """
    构建邻接矩阵  --- 图论
    :param vertex_num: 顶点数量
    :param skeleton: 关节点连接关系
    :param flip_pairs: 对称关节点关系
    :return: 顶点邻接矩阵
    """
    adj_matrix = np.zeros((vertex_num, vertex_num))
    # 下边这两行代码的效果是一样的
    for line in skeleton:
        adj_matrix[line] = 1
        adj_matrix[line[0], line[1]] = 1
    for pair in flip_pairs:
        adj_matrix[pair] = 1
        adj_matrix[pair[0], pair[1]] = 1

    return adj_matrix


def normalize_adj(adj):
    """
    归一化邻接矩阵
    :param adj: 邻接矩阵
    :return: 归一化之后的邻接矩阵
    """
    vertex_num = adj.shape[0]
    adj_self = adj + np.eye(vertex_num)
    eps = 1e-8
    D = np.diag(adj_self.sum(0)) + np.spacing(np.array(0))
    _D = 1 / np.sqrt(D)
    _D = _D * np.eye(vertex_num)
    normalized_adj = np.dot(np.dot(_D, adj_self), _D)
    return normalized_adj


def transform_joint_to_other_db(src_joint, src_name, dst_name):
    """
    将其他数据集关节点转换到coco数据集的关节点
    :param src_joint: 原数据集的关节点
    :param src_name: 原数据集的关节点的名字
    :param dst_name: 目标数据集关节点的名字
    :return: 目标数据解关节点坐标
    """
    src_joint_num = len(src_name)  # 17   任何数据集关键点  bs x 17 x 3
    dst_joint_num = len(dst_name)  # 17   COCO
    #
    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(src_joint_num):
        name = src_name[src_idx]
        if name in dst_name:
            dst_dix = dst_name.index(name)
            new_joint[dst_dix] = src_joint[src_idx]
    return new_joint


def world2cam(world_coord, R, t):
    """
    坐标系变化 --> 将世界坐标系转换到相机坐标系下
    :param world_coord: 世界坐标系的坐标
    :param R: 相机旋转矩阵
    :param t: 相机平移矩阵
    :return: 相机坐标系下的坐标
    """
    cam_coord = np.dot(R, world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
    return cam_coord


def cam2pixel(cam_coord, f, c):
    """
    坐标系变化 --> 将相机坐标系转换到像素坐标系
    :param cam_coord: 相机坐标系下的坐标
    :param f: 相机焦距
    :param c: 像素中心
    :return: 像素坐标系下的坐标
    """
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def pixel2cam():
    pass


def rigid_transform_3D(A, B):
    """
    这段代码实现了在三维空间中计算两组点集 A 和 B 之间的刚性变换（包括旋转和平移），使它们之间的平方误差最小化。具体步骤如下：
        计算两组点集 A 和 B 的质心（centroid_A 和 centroid_B）。
        计算 A 和 B 的协方差矩阵 H，即将 A 和 B 中心化后的点集的内积的平均值。
        对 H 进行奇异值分解（SVD），得到左奇异向量矩阵 U、奇异值矩阵 s 和右奇异向量矩阵 V。
        计算旋转矩阵 R，通过将 V 的转置乘以 U 的转置得到，确保旋转矩阵的行列式不为负值。
        计算尺度因子 c，用于调整旋转矩阵的尺度。
        计算平移向量 t，通过将 c 乘以 R 再乘以 A 的质心的负值，加上 B 的质心。
        返回尺度因子 c、旋转矩阵 R 和平移向量 t。
    需要注意的是，这段代码中假设了两组点集 A 和 B 具有相同的大小，且没有进行输入的验证。
    在使用时，请确保输入的点集格式正确，并根据需要进行适当的数据验证。
    :param A: 点集 A
    :param B: 点集 B
    :return: 尺度因子 c、旋转矩阵 R 和平移向量 t
    """
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)

    t = -np.dot(c * R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    """
    这个函数 rigid_align(A, B) 使用了之前定义的 rigid_transform_3D(A, B) 函数来计算将点集 A 对齐到点集 B 的刚体变换，并返回对齐后的点集 A。
    具体来说，这个函数实现了以下步骤：
        调用 rigid_transform_3D(A, B) 函数计算点集 A 到点集 B 的刚体变换参数，包括缩放因子 c、旋转矩阵 R 和平移向量 t。
        将点集 A 根据计算得到的刚体变换参数进行变换，得到对齐后的点集 A2。
        返回对齐后的点集 A2。
        这个函数主要用于将两个点云进行刚体配准，将一个点云对齐到另一个点云的空间位置和方向
    """
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c * R, np.transpose(A))) + t
    return A2


def rot6d_to_axis_angle(x):
    """
    用于将6D旋转表示转换为轴-角度表示
    过程:
        函数首先将输入的6D表示张量 x 重塑为形状 (batch_size, 3, 2)，
        然后从中提取出两个单位向量 a1 和 a2。接下来，函数对 a1 进行归一化处理得到单位向量 b1，然后计算第二个单位向量 b2，使其与 b1 垂直，
        并且与 a2 具有相同的方向。最后，函数利用 b1、b2 和它们的叉积计算出旋转矩阵 rot_mat，并将其转换为轴-角度表示 axis_angle。
    :param x: 形状为 (batch_size, 6) 的张量，表示一组旋转的6D表示，其中每个旋转由两个单位向量构成
    :return: 返回一个形状为 (batch_size, 3) 的张量，表示一组旋转的轴-角度表示，其中每个旋转由一个三维单位向量（轴）和一个标量（角度）组成
    """
    batch_size = x.shape[0]

    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix

    rot_mat = torch.cat([rot_mat, torch.zeros((batch_size, 3, 1)).cuda().float()], 2)  # 3x4 rotation matrix
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)  # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle


def denorm_joints(pose_out_img, body_bb2img_trans):
    """
    这段代码实现了一个函数 denorm_joints(pose_out_img, body_bb2img_trans)，用于将归一化后的关键点坐标还原为原始图像坐标系下的坐标。
    具体来说，这个函数的实现如下：
    将归一化后的关键点坐标 pose_out_img 中的 x 坐标除以 cfg.output_hm_shape[2]，然后乘以 cfg.input_img_shape[1]，以将 x 坐标转换为原始图像中的坐标。
    将归一化后的关键点坐标 pose_out_img 中的 y 坐标除以 cfg.output_hm_shape[1]，然后乘以 cfg.input_img_shape[0]，以将 y 坐标转换为原始图像中的坐标。
    将关键点坐标 [x, y] 扩展为齐次坐标 [x, y, 1]。
    将关键点坐标 pose_out_img 通过矩阵 body_bb2img_trans 进行仿射变换，得到关键点在原始图像中的坐标。
    最终，函数返回的 pose_out_img 中存储的是关键点在原始图像中的坐标。
    """
    pose_out_img[:, 0] = pose_out_img[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    pose_out_img[:, 1] = pose_out_img[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    pose_out_img_xy1 = np.concatenate((pose_out_img[:, :2], np.ones_like(pose_out_img[:, :1])), 1)
    pose_out_img[:, :2] = np.dot(body_bb2img_trans, pose_out_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]


def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    """
    这段代码实现了一个函数 convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height)，用于将在裁剪图像坐标系下预测的相机参数转换为原始图像坐标系下的相机参数。
    具体来说，这个函数的实现如下：
        从参数 bbox 中提取出裁剪图像的中心点坐标 cx, cy 和裁剪框的高度 h。
        计算原始图像的中心点坐标 hw, hh，即原始图像宽度和高度的一半。
        将相机参数 cam 中的缩放因子 sx, sy 分别乘以裁剪图像宽度和高度与裁剪框高度的比值，以将缩放因子转换为原始图像坐标系下的缩放因子。
        计算相机参数 cam 中的平移参数 tx, ty，通过将裁剪框中心点坐标减去原始图像中心点坐标，并除以原始图像中心点坐标的一半，再除以相应的缩放因子，最后加上相机参数中的平移参数。
        将转换后的缩放因子 sx, sy 和平移参数 tx, ty 组合成原始图像坐标系下的相机参数 orig_cam，并返回。
    这个函数通常用于在姿态估计等任务中，将在裁剪图像坐标系下预测的相机参数转换为原始图像坐标系下，以便更好地理解和可视化姿态估计结果。
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    """
    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:, 0] * (1. / (img_width / h))
    sy = cam[:, 0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:, 1]
    ty = ((cy - hh) / hh / sy) + cam[:, 2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam
