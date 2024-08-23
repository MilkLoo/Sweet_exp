import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from PIL import Image


# 假设 keypoint 是一个包含关键点位置的数组，形状为 (N, 2)，N 表示关键点的数量
# 假设 image 是原始图像

def generate_heatmap(image_path, keypoints, sigma=10):
    """
    生成并显示包含关键点热力图的图像。

    :param image_path: 原始图像路径。
    :param keypoints: 关键点的列表，每个关键点是(x, y)坐标的元组。
    :param sigma: 用于生成热力图的高斯模糊的标准差。
    """
    # 读取原始图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # 创建一个空的图像用于热力图
    heatmap = np.zeros((height, width), dtype=np.float32)

    # 在每个关键点位置添加高斯点
    for x, y in keypoints:
        heatmap[int(y), int(x)] = 1

    # 应用高斯模糊生成热力图
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)
    heatmap /= np.max(heatmap)  # 归一化热力图

    # 将热力图应用到原始图像
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    result_image = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)

    # 可视化原始图像和热力图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')

    # 使用matplotlib显示结果
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Heatmap with Key Points')
    plt.axis('off')  # 不显示坐标轴
    plt.show()


if __name__ == "__main__":
    ori_image_name = "057.jpg"  # 可以修改
    ori_image_path = "/home/ly/yxc_exp_smpl/2D_pose_generator_tool/core_code/images/057.jpg"
    keypoint_2d_data_path = "/home/ly/yxc_exp_smpl/2D_pose_generator_tool/core_code/2d_pose.json"
    # keypoint_2d_data_path = "/home/ly/yxc_exp_smpl/2D_pose_estimation_tool/2d_pose_transformer/2d_pose_est.json"
    with open(keypoint_2d_data_path, "r") as data_file:
        data = json.load(data_file)
    # ori_keypoints = np.array(data[ori_image_name])[:, :, :2].view((17, 2))
    ori_keypoints = np.array(data[ori_image_name])[:, :, :2].reshape((17,2))
    generate_heatmap(ori_image_path, ori_keypoints)
