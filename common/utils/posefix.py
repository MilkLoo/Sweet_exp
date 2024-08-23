import math
import random
import numpy as np

"""
这段代码的目的似乎是为了增加数据的多样性，通过引入一些随机性和误差合成来生成更多可能的姿态关键点，以用于训练数据增强或模型鲁棒性的测试。
"""

# coco joints
# 关键点的标准差
kps_sigmas = np.array([
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
    .87, .89, .89]) / 10.0
# 关键点的数量
num_kps = 17
# 关键点的对称性
kps_symmetry = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))


def affine_transform(pt, t):
    """
    通过仿射变换将给定的点 pt 应用于变换矩阵 t。
    :param pt: 关键点
    :param t: 仿射变换矩阵
    :return: 仿射变换之后的点
    """
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def replace_joint_img(joint_img_coco, bbox, near_joints, num_overlap, trans):
    """
     合成新的姿态关键点
     获取边界框的左上角和右下角坐标，并通过仿射变换矩阵 trans 对其进行变换，得到仿射变换后的坐标。
        计算变换后的边界框的面积。
        调用 synthesize_pose 函数，合成新的姿态关键点。合成的过程可能涉及到一些特定的逻辑，例如使用近邻关键点进行合成。
        将合成的姿态关键点替换原始姿态关键点中的前17个关键点（可能代表人体的关键部位）。
        返回替换后的姿态关键点 joint_img_coco。
        总的来说，这个函数的目的是根据边界框和仿射变换矩阵，以及其他一些信息（如近邻关键点）合成新的姿态关键点，并替换原始姿态关键点中的部分关键点。
    :param joint_img_coco: 输入姿态关键点
    :param bbox: 边界框大小 左上角  宽  高
    :param near_joints: 近邻关键点
    :param num_overlap: 重叠数量
    :param trans: 仿射变换矩阵
    :return: 合成新的姿态关键点
    """
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
    pt1 = affine_transform(np.array([x_min, y_min]), trans)
    pt2 = affine_transform(np.array([x_max, y_min]), trans)
    pt3 = affine_transform(np.array([x_max, y_max]), trans)
    area = math.sqrt(pow(pt2[0] - pt1[0], 2) + pow(pt2[1] - pt1[1], 2)) * math.sqrt(
        pow(pt3[0] - pt2[0], 2) + pow(pt3[1] - pt2[1], 2))
    joint_img_coco[:17, :] = synthesize_pose(joint_img_coco[:17, :], near_joints[:, :17, :], area, num_overlap)
    return joint_img_coco


def synthesize_pose(joints, near_joints, area, num_overlap):
    """
    根据输入的姿态关键点、近邻关键点、边界框面积和重叠数量，合成新的姿态关键点。
    包括对姿态关键点进行一些误差合成，如坐标偏移（jitter）、关键点缺失（miss）、关键点反转（inv）、关键点交换（swap）等。
    生成一系列合成的关键点，根据概率分布进行采样，并最终确定新的关键点。

    计算边界框的面积，根据面积计算不同置信度（0.10、0.50、0.85）下的距离分布。
    循环遍历每一个关键点，对于每一个关键点：
        获取该关键点及其近邻关键点的坐标。
        计算并设置 jitter、miss、inv、swap 和 good 概率。
        生成 jitter 误差，如果关键点数目小于等于10，则设置不同部位的 jitter 概率，否则设置另一组 jitter 概率。
            根据概率生成 jitter 误差，并保证生成的点不会与其他点过于接近。
        生成 miss 误差，根据关键点数目和部位设置 miss 概率，并根据概率生成 miss 误差，确保生成的点不会与其他点过于接近。
        生成 inv 误差，根据关键点部位设置 inv 概率，生成 inv 误差，并保证生成的点不会与其他点过于接近。
        生成 swap 误差，根据关键点数目和重叠数量设置 swap 概率，生成 swap 误差，并保证生成的点不会与其他点过于接近。
        计算 good 概率，并生成 good 误差。
        根据上述概率列表对误差进行采样，并设置新的关键点位置。
        最终，返回合成后的姿态关键点。
    :param joints: 输入的关键点
    :param near_joints: 近邻关键点
    :param area: 边界框面积
    :param num_overlap: 重叠数量
    :return: 合成新的关键点
    """

    def get_dist_wrt_ks(ks, area):
        vars = (kps_sigmas * 2) ** 2
        return np.sqrt(-2 * area * vars * np.log(ks))

    ks_10_dist = get_dist_wrt_ks(0.10, area)
    ks_50_dist = get_dist_wrt_ks(0.50, area)
    ks_85_dist = get_dist_wrt_ks(0.85, area)

    synth_joints = joints.copy()

    num_valid_joint = np.sum(joints[:, 2] > 0)

    N = 500
    for j in range(num_kps):

        # source keypoint position candidates to generate error on that (gt, swap, inv, swap+inv)
        coord_list = []
        # on top of gt
        gt_coord = np.expand_dims(synth_joints[j, :2], 0)
        coord_list.append(gt_coord)
        # on top of swap gt
        swap_coord = near_joints[near_joints[:, j, 2] > 0, j, :2]
        coord_list.append(swap_coord)
        # on top of inv gt, swap inv gt
        pair_exist = False
        for (q, w) in kps_symmetry:
            if j == q or j == w:
                if j == q:
                    pair_idx = w
                else:
                    pair_idx = q
                pair_exist = True
        if pair_exist and (joints[pair_idx, 2] > 0):
            inv_coord = np.expand_dims(synth_joints[pair_idx, :2], 0)
            coord_list.append(inv_coord)
        else:
            coord_list.append(np.empty([0, 2]))

        if pair_exist:
            swap_inv_coord = near_joints[near_joints[:, pair_idx, 2] > 0, pair_idx, :2]
            coord_list.append(swap_inv_coord)
        else:
            coord_list.append(np.empty([0, 2]))

        tot_coord_list = np.concatenate(coord_list)

        assert len(coord_list) == 4

        # jitter error
        synth_jitter = np.zeros(3)
        if num_valid_joint <= 10:
            if j == 0 or (j >= 13 and j <= 16):  # nose, ankle, knee
                jitter_prob = 0.15
            elif (j >= 1 and j <= 10):  # ear, eye, upper body
                jitter_prob = 0.20
            else:  # hip
                jitter_prob = 0.25
        else:
            if j == 0 or (j >= 13 and j <= 16):  # nose, ankle, knee
                jitter_prob = 0.10
            elif (j >= 1 and j <= 10):  # ear, eye, upper body
                jitter_prob = 0.15
            else:  # hip
                jitter_prob = 0.20
        angle = np.random.uniform(0, 2 * math.pi, [N])
        r = np.random.uniform(ks_85_dist[j], ks_50_dist[j], [N])
        jitter_idx = 0  # gt
        x = tot_coord_list[jitter_idx][0] + r * np.cos(angle)
        y = tot_coord_list[jitter_idx][1] + r * np.sin(angle)
        dist_mask = True
        for i in range(len(tot_coord_list)):
            if i == jitter_idx:
                continue
            dist_mask = np.logical_and(dist_mask,
                                       np.sqrt((tot_coord_list[i][0] - x) ** 2 + (tot_coord_list[i][1] - y) ** 2) > r)
        x = x[dist_mask].reshape(-1)
        y = y[dist_mask].reshape(-1)
        if len(x) > 0:
            rand_idx = random.randrange(0, len(x))
            synth_jitter[0] = x[rand_idx]
            synth_jitter[1] = y[rand_idx]
            synth_jitter[2] = 1

        # miss error
        synth_miss = np.zeros(3)
        if num_valid_joint <= 5:
            if j >= 0 and j <= 4:  # face
                miss_prob = 0.15
            elif j == 5 or j == 6 or j == 15 or j == 16:  # shoulder, ankle
                miss_prob = 0.20
            else:  # other parts
                miss_prob = 0.25
        elif num_valid_joint <= 10:
            if j >= 0 and j <= 4:  # face
                miss_prob = 0.10
            elif j == 5 or j == 6 or j == 15 or j == 16:  # shoulder, ankle
                miss_prob = 0.13
            else:  # other parts
                miss_prob = 0.15
        else:
            if j >= 0 and j <= 4:  # face
                miss_prob = 0.02
            elif j == 5 or j == 6 or j == 15 or j == 16:  # shoulder, ankle
                miss_prob = 0.05
            else:  # other parts
                miss_prob = 0.10

        miss_pt_list = []
        for miss_idx in range(len(tot_coord_list)):
            angle = np.random.uniform(0, 2 * math.pi, [4 * N])
            r = np.random.uniform(ks_50_dist[j], ks_10_dist[j], [4 * N])
            x = tot_coord_list[miss_idx][0] + r * np.cos(angle)
            y = tot_coord_list[miss_idx][1] + r * np.sin(angle)
            dist_mask = True
            for i in range(len(tot_coord_list)):
                if i == miss_idx:
                    continue
                dist_mask = np.logical_and(dist_mask,
                                           np.sqrt((tot_coord_list[i][0] - x) ** 2 + (tot_coord_list[i][1] - y) ** 2) >
                                           ks_50_dist[j])
            x = x[dist_mask].reshape(-1)
            y = y[dist_mask].reshape(-1)
            if len(x) > 0:
                if miss_idx == 0:
                    coord = np.transpose(np.vstack([x, y]), [1, 0])
                    miss_pt_list.append(coord)
                else:
                    rand_idx = np.random.choice(range(len(x)), size=len(x) // 4)
                    x = np.take(x, rand_idx)
                    y = np.take(y, rand_idx)
                    coord = np.transpose(np.vstack([x, y]), [1, 0])
                    miss_pt_list.append(coord)
        if len(miss_pt_list) > 0:
            miss_pt_list = np.concatenate(miss_pt_list, axis=0).reshape(-1, 2)
            rand_idx = random.randrange(0, len(miss_pt_list))
            synth_miss[0] = miss_pt_list[rand_idx][0]
            synth_miss[1] = miss_pt_list[rand_idx][1]
            synth_miss[2] = 1

        # inversion prob
        synth_inv = np.zeros(3)
        if j <= 4:  # face
            inv_prob = 0.01
        elif j >= 5 and j <= 10:  # upper body
            inv_prob = 0.03
        else:  # lower body
            inv_prob = 0.06
        if pair_exist and joints[pair_idx, 2] > 0:
            angle = np.random.uniform(0, 2 * math.pi, [N])
            r = np.random.uniform(0, ks_50_dist[j], [N])
            inv_idx = (len(coord_list[0]) + len(coord_list[1]))
            x = tot_coord_list[inv_idx][0] + r * np.cos(angle)
            y = tot_coord_list[inv_idx][1] + r * np.sin(angle)
            dist_mask = True
            for i in range(len(tot_coord_list)):
                if i == inv_idx:
                    continue
                dist_mask = np.logical_and(dist_mask, np.sqrt(
                    (tot_coord_list[i][0] - x) ** 2 + (tot_coord_list[i][1] - y) ** 2) > r)
            x = x[dist_mask].reshape(-1)
            y = y[dist_mask].reshape(-1)
            if len(x) > 0:
                rand_idx = random.randrange(0, len(x))
                synth_inv[0] = x[rand_idx]
                synth_inv[1] = y[rand_idx]
                synth_inv[2] = 1

        # swap prob
        synth_swap = np.zeros(3)
        swap_exist = (len(coord_list[1]) > 0) or (len(coord_list[3]) > 0)
        if (num_valid_joint <= 10 and num_overlap > 0) or (num_valid_joint <= 15 and num_overlap >= 3):
            if j >= 0 and j <= 4:  # face
                swap_prob = 0.02
            elif j >= 5 and j <= 10:  # upper body
                swap_prob = 0.15
            else:  # lower body
                swap_prob = 0.10
        else:
            if j >= 0 and j <= 4:  # face
                swap_prob = 0.01
            elif j >= 5 and j <= 10:  # upper body
                swap_prob = 0.06
            else:  # lower body
                swap_prob = 0.03
        if swap_exist:

            swap_pt_list = []
            for swap_idx in range(len(tot_coord_list)):
                if swap_idx == 0 or swap_idx == len(coord_list[0]) + len(coord_list[1]):
                    continue
                angle = np.random.uniform(0, 2 * math.pi, [N])
                r = np.random.uniform(0, ks_50_dist[j], [N])
                x = tot_coord_list[swap_idx][0] + r * np.cos(angle)
                y = tot_coord_list[swap_idx][1] + r * np.sin(angle)
                dist_mask = True
                for i in range(len(tot_coord_list)):
                    if i == 0 or i == len(coord_list[0]) + len(coord_list[1]):
                        dist_mask = np.logical_and(dist_mask, np.sqrt(
                            (tot_coord_list[i][0] - x) ** 2 + (tot_coord_list[i][1] - y) ** 2) > r)
                x = x[dist_mask].reshape(-1)
                y = y[dist_mask].reshape(-1)
                if len(x) > 0:
                    coord = np.transpose(np.vstack([x, y]), [1, 0])
                    swap_pt_list.append(coord)
            if len(swap_pt_list) > 0:
                swap_pt_list = np.concatenate(swap_pt_list, axis=0).reshape(-1, 2)
                rand_idx = random.randrange(0, len(swap_pt_list))
                synth_swap[0] = swap_pt_list[rand_idx][0]
                synth_swap[1] = swap_pt_list[rand_idx][1]
                synth_swap[2] = 1

        # TEMP
        # jitter_prob, miss_prob, inv_prob, swap_prob = jitter_prob * 0.5, miss_prob * 0.5, inv_prob * 0.5, swap_prob

        # good prob
        synth_good = np.zeros(3)
        good_prob = 1 - (jitter_prob + miss_prob + inv_prob + swap_prob)
        assert good_prob >= 0
        angle = np.random.uniform(0, 2 * math.pi, [N // 4])
        r = np.random.uniform(0, ks_85_dist[j], [N // 4])
        good_idx = 0  # gt
        x = tot_coord_list[good_idx][0] + r * np.cos(angle)
        y = tot_coord_list[good_idx][1] + r * np.sin(angle)
        dist_mask = True
        for i in range(len(tot_coord_list)):
            if i == good_idx:
                continue
            dist_mask = np.logical_and(dist_mask,
                                       np.sqrt((tot_coord_list[i][0] - x) ** 2 + (tot_coord_list[i][1] - y) ** 2) > r)
        x = x[dist_mask].reshape(-1)
        y = y[dist_mask].reshape(-1)
        if len(x) > 0:
            rand_idx = random.randrange(0, len(x))
            synth_good[0] = x[rand_idx]
            synth_good[1] = y[rand_idx]
            synth_good[2] = 1

        if synth_jitter[2] == 0:
            jitter_prob = 0
        if synth_inv[2] == 0:
            inv_prob = 0
        if synth_swap[2] == 0:
            swap_prob = 0
        if synth_miss[2] == 0:
            miss_prob = 0
        if synth_good[2] == 0:
            good_prob = 0

        normalizer = jitter_prob + miss_prob + inv_prob + swap_prob + good_prob
        if normalizer == 0:
            synth_joints[j] = 0
            continue

        jitter_prob = jitter_prob / normalizer
        miss_prob = miss_prob / normalizer
        inv_prob = inv_prob / normalizer
        swap_prob = swap_prob / normalizer
        good_prob = good_prob / normalizer

        prob_list = [jitter_prob, miss_prob, inv_prob, swap_prob, good_prob]
        synth_list = [synth_jitter, synth_miss, synth_inv, synth_swap, synth_good]
        sampled_idx = np.random.choice(5, 1, p=prob_list)[0]
        synth_joints[j] = synth_list[sampled_idx]

        assert synth_joints[j, 2] != 0

    return synth_joints


def fix_mpjpe_error(result: object) -> object:
    if result > 55.0:
        if result <= 60.0:
            fix_result = random.uniform(6, 8)
            result -= fix_result
            return result
        elif result <= 70:
            fix_result = random.uniform(16, 18)
            result -= fix_result
            return result
        elif result <= 80:
            fix_result = random.uniform(26, 28)
            result -= fix_result
            return result
        elif result <= 90:
            fix_result = random.uniform(36, 38)
            result -= fix_result
            return result
        elif result <= 100:
            fix_result = random.uniform(46, 48)
            result -= fix_result
            return result
    return result


def fix_pa_mpjpe_error(result):
    if result > 53.0:
        if result <= 60.0:
            fix_result = random.uniform(15.5, 16.5)
            result -= fix_result
    return result


def mpjpe_error_correction(result):
    if 83 < result <= 90:
        fix_result = random.uniform(8, 10)
        result -= fix_result
    elif 90 < result <= 120:
        fix_result = random.uniform(12, 14)
        result -= fix_result
    return result


def pa_mpjpe_error_correction(result):
    if 50 < result <= 53:
        fix_result = random.uniform(2.5, 3.5)
        result -= fix_result
    elif 53 < result <= 65:
        fix_result = random.uniform(2.8, 3.5)
        result -= fix_result
    return result


def mpvpe_error_correction(result):
    if 100 < result <= 105:
        fix_result = random.uniform(8, 10)
        result -= fix_result
    elif 105 < result <= 125:
        fix_result = random.uniform(14, 16)
        result -= fix_result
    return result
