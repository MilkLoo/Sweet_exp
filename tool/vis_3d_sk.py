import os
# import cv2
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import trimesh
# import pyrender
from common.utils.vis import vis_3d_skeleton
import json

os.environ['PYOPENGL_PLATFORM'] = 'egl'

if __name__ == "__main__":
    # file_path = "/home/ly/yxc_exp_smpl/tool/3d_keypoint_data/3d_joint_data.json"
    file_path = "/home/ly/yxc_exp_smpl/tool/3d_keypoint_data/obj_to_sk.json"
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    # 30 个关键点的名称
    joints_name = (
        'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe',
        'R_Toe',
        'Neck', 'L_Thorax', 'R_Thorax',
        'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose',
        'L_Eye',
        'R_Eye', 'L_Ear', 'R_Ear', 'Head_top')
    skeleton = (
        (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
        (17, 19),
        (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 24), (24, 15), (24, 25),
        (24, 26),
        (25, 27), (26, 28), (24, 29))

    graph_joint_name = ("Pelvis", "l_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
                        "Neck", "Head_top", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
                        "L_Wrist", "R_Wrist")

    graph_skeleton = ((0, 1), (1, 3), (3, 5),
                      (0, 2), (2, 4), (4, 6),
                      (0, 7), (7, 8), (7, 9), (9, 11), (11, 13),
                      (7, 10), (10, 12), (12, 14))
    # 需要自己设置  以及  坐标那些要 哪些 不要
    vis_3d_keypoint = np.ones((30, 1), dtype=np.float32)
    # 这是一般的 3D 骨架显示
    # for i in range(len(joints_name)):
    #     if i in (3, 6, 10, 11, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29):
    #         vis_3d_keypoint[i, 0] = 0.
    # skeleton_fix = (
    #     (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 12), (3, 6), (6, 9), (9, 14), (14, 17),
    #     (17, 19),
    #     (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 24), (24, 15), (24, 25),
    #     (24, 26),
    #     (25, 27), (26, 28), (24, 29))
    # 这是画结构图需要的
    for i in range(len(joints_name)):
        if i in (3, 6, 9, 10, 11, 13, 14, 15, 22, 23, 24, 25, 26, 27, 28):
            vis_3d_keypoint[i, 0] = 0.
    skeleton_fix = (
        (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 12), (12, 29), (3, 6), (6, 9), (9, 14),
        (14, 17), (12, 16), (12, 17),
        (17, 19),
        (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 24), (24, 15), (24, 25),
        (24, 26),
        (25, 27), (26, 28), (24, 29))
    vis_3d_skeleton(np.array(data["people_0"]), vis_3d_keypoint, skeleton_fix)
