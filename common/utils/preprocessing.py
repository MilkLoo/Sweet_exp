import numpy as np
import cv2
import random
from config import cfg
import math
from PIL import Image
import io
from einops import repeat


def process_bbox(bbox, img_width, img_height, is_3dpw_test=False):  # 这个公式之后可能要改
    """
    整理锚框  两种方式 --> 1.接受输入的自定义的宽高 2. 接受比例形式
    首先，函数接受一个边界框 bbox，以及图像的宽度和高度 img_width、img_height。
        它还接受一个布尔值 is_3dpw_test，用于指示是否在处理 3DPW 测试数据。根据情况，边界框可能以不同的方式进行处理。
    接着，函数从边界框中提取出左上角和右下角的坐标，并确保这些坐标在图像范围内。
            如果 is_3dpw_test 为真，则将边界框坐标存储为浮点数类型。
            如果不是，且边界框的宽度和高度大于零，则将边界框坐标存储为浮点数类型；否则返回 None。
    其次，如果 is_3dpw_test 为真，则将比例因子 scale 设置为 1.1；
        否则，随机生成一个值并添加到 1.0 上，以确定新的边界框大小。然后，根据图像的宽高比，调整边界框的宽度和高度。
    最后，重新计算边界框的中心点坐标，并根据新的宽度和高度调整左上角的坐标。返回更新后的边界框。
    :param bbox: 输入的锚框
    :param img_width: 图片的宽
    :param img_height: 图片的高
    :param is_3dpw_test: 是不是3dpw数据集的标志位
    :return: 处理好的锚框
    """
    # first
    x, y, w, h = bbox
    x_1 = np.max((0, x))
    y_1 = np.max((0, y))
    x_2 = np.min((img_width - 1, x_1 + np.max((0, w - 1))))
    y_2 = np.min((img_height - 1, y_1 + np.max((0, h - 1))))
    if is_3dpw_test:
        bbox = np.array([x_1, y_1, x_2 - x_1, y_2 - y_1], dtype=np.float32)
    else:
        if w * h > 0 and x_2 >= x_1 and y_2 >= y_1:
            bbox = np.array([x_1, y_1, x_2 - x_1, y_2 - y_1], dtype=np.float32)
        else:
            return None
    # second
    if is_3dpw_test:
        scale = 1.1
    else:
        scale = np.random.rand() / 5 + 1  # 1.0~1.2

    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = cfg.input_img_shape[1] / cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * scale
    bbox[3] = h * scale
    bbox[0] = c_x - bbox[2] / 2
    bbox[1] = c_y - bbox[3] / 2

    return bbox


def load_img(path, order="RGB"):
    """
    以rgb的形式加载图片
    :param path: 图片路径
    :param order: 形式
    :return: 加载好的图片
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError('Fail to read %s' % path)
    if order == "RGB":
        img = img[:, :, ::-1].copy()

    img = img.astype(np.float32)
    return img


def rotate_2d(pt_2d, rot_rad):
    """
     旋转之后的坐标
    :param pt_2d: 旋转之前的坐标点
    :param rot_rad: 旋转的弧度
    :return: 旋转之后的坐标
    """
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = y * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def get_aug_config(exclude_flip):
    """
    生成随机数据增强的因子
    :param exclude_flip: 翻转标志位
    :return: 随机数据增强的因子
    """
    scale_factor = 0.25
    rot_scale = 30
    color_factor = 0.2

    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0, 2.0) * rot_scale if random.random() <= 0.6 else 0
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_factor = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    if exclude_flip:
        do_flip = False
    else:
        do_flip = random.random() <= 0.5
    return scale, rot, color_factor, do_flip


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    """
    生成仿射变换矩阵
    首先，函数接受源图像中心点的坐标 c_x 和 c_y，以及源图像的宽度 src_width 和高度 src_height。
        还接受目标图像的宽度 dst_width 和高度 dst_height，以及缩放比例 scale 和旋转角度 rot。
    然后，计算源图像的宽度和高度经过缩放后的实际宽度 src_w 和高度 src_h。并定义源图像的中心点坐标 src_center。
    接着，将旋转角度 rot 转换为弧度制，并根据旋转角度计算源图像的下方向向量 src_down_dir 和右方向向量 src_right_dir。
    同样地，计算目标图像的宽度和高度，定义目标图像的中心点坐标 dst_center，以及下方向向量 dst_down_dir 和右方向向量 dst_right_dir。
    创建源图像和目标图像的坐标点数组 src 和 dst，分别存储了图像中心点、中心点下方的点和中心点右方的点。
    如果需要逆变换（从目标图像到源图像），则调用 cv2.getAffineTransform 函数，传入目标图像到源图像的坐标对。否则，传入源图像到目标图像的坐标对。
    将生成的仿射变换矩阵转换为 32 位浮点数并返回。
    这个函数主要用于在图像处理中进行局部区域的仿射变换，例如在目标检测或图像配准等任务中
    :param c_x: 原始图像的中心点 x
    :param c_y: 原始图像的中心点 x
    :param src_width: 原始图像的宽
    :param src_height: 原始图像的高
    :param dst_width: 目标图像的宽
    :param dst_height: 目标图像的高
    :param scale: 缩放因子
    :param rot: 旋转因子
    :param inv: 逆变换标志位
    :return: 仿射变换的矩阵 2 x 3
    """
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    rot_rad = np.pi * rot / 180  # 弧度制
    src_down_dir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_right_dir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_down_dir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_right_dir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_down_dir
    src[2, :] = src_center + src_right_dir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_down_dir
    dst[2, :] = dst_center + dst_right_dir

    if inv:
        # 生成仿射变换的变换矩阵，需要原始图像和目标图像的平行四边形的三个点坐标（左上角，右上角，左下角）
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def generate_patch_image(cv_img, bbox, scale, rot, do_flip, out_shape):
    """
    生成变换之后的批量图片
    函数接受原始图像 cv_img、边界框 bbox、缩放比例 scale、旋转角度 rot、是否翻转 do_flip 和输出图像形状 out_shape。
    首先，复制原始图像以防止对其进行更改。
    然后，获取原始图像的高度 img_height、宽度 img_width 和通道数 img_channels，
        以及边界框的中心点坐标 bb_c_x 和 bb_c_y，宽度 bb_width 和高度 bb_height。
    如果需要翻转图像，则将图像沿着水平方向翻转，并相应地更新边界框的中心点坐标 bb_c_x。
    调用 gen_trans_from_patch_cv 函数生成仿射变换矩阵 trans，该矩阵用于将原始图像中的边界框裁剪、缩放、旋转到目标图像中。
    使用 cv2.warpAffine 函数将原始图像 img 根据仿射变换矩阵 trans 进行变换，生成裁剪、缩放、旋转后的图像补丁 img_patch。
    将图像补丁转换为 32 位浮点数类型。
    根据逆变换矩阵 inv_trans 将边界框的中心点从目标图像坐标系转换回原始图像坐标系。
    返回图像补丁 img_patch，正向变换矩阵 trans 和逆变换矩阵 inv_trans。
    这个函数通常用于数据增强，特别是在目标检测任务中，用于生成经过处理的图像补丁，以提高模型的鲁棒性和泛化能力
    :param cv_img: 图片
    :param bbox: 锚框大小
    :param scale: 缩放因子
    :param rot: 旋转因子
    :param do_flip: 翻转标志位
    :param out_shape: 输出大小
    :return: 生成变换之后的批量图片，仿射变换矩阵，仿射逆变换矩阵
    """
    img = cv_img.copy()
    img_height, img_width, img_channels = img.shape
    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1
    # 边界框随着图像大小 变换
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot,
                                        inv=True)
    return img_patch, trans, inv_trans


def augmentation(img, bbox, data_split, exclude_flip=False):
    """
     图片的数据增强
     函数接受输入图像 img、边界框 bbox、数据分割类型 data_split 和是否排除翻转增强 exclude_flip。
    如果数据分割类型为训练集 "train"，则调用 get_aug_config 函数获取训练集数据增强的配置参数：缩放比例 scale、旋转角度 rot、
        颜色缩放因子 color_scale 和是否翻转 do_flip。否则，设置默认参数：缩放比例为 1.0，旋转角度为 0.0，颜色缩放因子为全为 1，不进行翻转。
    调用 generate_patch_image 函数生成经过裁剪、缩放、旋转和翻转处理后的图像补丁 img，并获取正向变换矩阵 trans 和逆变换矩阵 inv_trans，
        目标图像形状为 cfg.input_img_shape。
    将生成的图像补丁 img 按颜色缩放因子进行裁剪，将像素值限制在 [0, 255] 范围内。
    返回处理后的图像补丁 img、正向变换矩阵 trans、逆向变换矩阵 inv_trans、旋转角度 rot 和是否翻转 do_flip。
    这个函数通常用于训练集和测试集的数据增强操作，在训练集上使用多种增强方法来增加数据的多样性，提高模型的泛化能力
    :param img: 原始图片
    :param bbox: 锚框大小
    :param data_split: 数据分割形式 训练集 -- 测试集
    :param exclude_flip: 包括翻转标志位
    :return: 数据增广之后的图像
    """
    if data_split == "train":
        scale, rot, color_scale, do_flip = get_aug_config(exclude_flip, )
    else:
        scale, rot, color_scale, do_flip = 1.0, 0.0, np.array([1, 1, 1]), False

    img, trans, inv_trans = generate_patch_image(img, bbox, scale, rot, do_flip, cfg.input_img_shape)
    img = np.clip(img * color_scale[None, None, :], 0, 255)
    return img, trans, inv_trans, rot, do_flip


def get_bbox(joint_img, joint_valid):
    """
    这段代码用于从检测到的人体关键点中获取包围框（bounding box）。让我们逐行解释：
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]:
                这行代码将检测到的人体关键点的 x 和 y 坐标分别提取出来，保存到变量 x_img 和 y_img 中。
                假设 joint_img 是一个二维数组，每行代表一个关键点，第一列是 x 坐标，第二列是 y 坐标。
    x_img = x_img[joint_valid == 1]; y_img = y_img[joint_valid == 1];:
                这行代码根据关键点的有效性（由 joint_valid 数组指示）对坐标进行过滤，只保留有效的关键点坐标。
                假设 joint_valid 是一个与 joint_img 形状相同的数组，其中每个元素表示对应关键点的有效性（例如，1 表示有效，0 表示无效）。
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);:
                这几行代码计算了有效关键点坐标的最小和最大值，从而得到未经缩放的包围框的左上角和右下角坐标。
    x_center = (xmin + xmax) / 2.; width = xmax - xmin;:
                这行代码计算了包围框的水平中心位置 x_center 和宽度 width。
    xmin = x_center - 0.5 * width * 1.2; xmax = x_center + 0.5 * width * 1.2:
                这两行代码根据给定的缩放因子（这里是 1.2）扩展了包围框的水平尺寸，以确保包围框能够充分覆盖关键点。
    y_center = (ymin + ymax) / 2.; height = ymax - ymin;
    ymin = y_center - 0.5 * height * 1.2; ymax = y_center + 0.5 * height * 1.2:
                类似于水平方向，这几行代码计算了包围框的垂直中心位置、高度，并根据缩放因子扩展了包围框的垂直尺寸。
    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32):
                最后，这行代码将左上角坐标、宽度和高度组合成一个一维数组，并将其转换为 np.float32 类型，得到最终的包围框。
    综合起来，这段代码的作用是从检测到的人体关键点中获取一个稍微扩展的包围框，以确保关键点的完整覆盖。
    :param joint_img: 人体关键点坐标
    :param joint_valid: 人体关键点的置信度
    :return: 获取一个稍微扩展的包围框，以确保关键点的完整覆盖。
    """
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1]
    y_img = y_img[joint_valid == 1]
    xmin = min(x_img)
    ymin = min(y_img)
    xmax = max(x_img)
    ymax = max(y_img)

    x_center = (xmin + xmax) / 2.
    width = xmax - xmin
    xmin = x_center - 0.5 * width * 1.2
    xmax = x_center + 0.5 * width * 1.2

    y_center = (ymin + ymax) / 2.
    height = ymax - ymin
    ymin = y_center - 0.5 * height * 1.2
    ymax = y_center + 0.5 * height * 1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def load_img_from_lmdb():
    pass


def compute_iou(src_roi, dst_roi):
    """
    代码定义了一个计算两个矩形区域之间IoU
    功能:
        计算两个矩形区域在x和y方向上的交叠部分的左上角和右下角坐标（x_min, y_min, x_max, y_max）
        计算交叠区域的面积（interArea）
        计算两个矩形区域各自的面积（boxAArea, boxBArea）
        计算IoU（Intersection over Union）：IoU = interArea / (sumArea - interArea + 1e-5)
    :param src_roi: 表示源矩形区域的Numpy数组，其形状为 (N, 4)，其中N是矩形数量，每个矩形由 (x, y, width, height) 描述
    :param dst_roi: 表示目标矩形区域的Numpy数组，其形状同样为 (N, 4)
    :return: 返回一个包含N个IoU值的Numpy数组，对应于输入的N个矩形
    """
    # IoU calculate with GTs
    x_min = np.maximum(dst_roi[:, 0], src_roi[:, 0])
    y_min = np.maximum(dst_roi[:, 1], src_roi[:, 1])
    x_max = np.minimum(dst_roi[:, 0] + dst_roi[:, 2], src_roi[:, 0] + src_roi[:, 2])
    y_max = np.minimum(dst_roi[:, 1] + dst_roi[:, 3], src_roi[:, 1] + src_roi[:, 3])

    interArea = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)

    boxAArea = dst_roi[:, 2] * dst_roi[:, 3]
    boxBArea = np.tile(src_roi[:, 2] * src_roi[:, 3], (len(dst_roi), 1))
    sumArea = boxAArea + boxBArea

    iou = interArea / (sumArea - interArea + 1e-5)

    return iou
