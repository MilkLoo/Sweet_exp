import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import trimesh
import pyrender

os.environ['PYOPENGL_PLATFORM'] = 'egl'


def vis_bbox(img, bbox, alpha=1):
    """
     这个函数用于在图像上绘制矩形边界框（bbox）并返回带有边界框的图像
     功能：
         将输入图像 img 复制到新的图像 kp_mask 中。
         将边界框的四个边绘制在 kp_mask 上，线条颜色为蓝色 (255, 255, 0)，线条粗细为1像素。
         使用 cv2.addWeighted 函数将原始图像 img 与带有边界框的 kp_mask 进行混合，以生成带有透明度的边界框图像。
    :param img: 输入图像
    :param bbox: 边界框坐标，格式为 [x, y, w, h]（左上角坐标 (x, y)，宽度 w，高度 h）。
    :param alpha: 边界框的透明度，介于 0 和 1 之间。
    :return: 图像上绘制一个矩形边界框，可以通过调整 alpha 参数来控制边界框的透明度。
    """
    kp_mask = np.copy(img)
    bbox = bbox.astype(np.int32)  # x, y, w, h

    b1 = bbox[0], bbox[1]
    b2 = bbox[0] + bbox[2], bbox[1]
    b3 = bbox[0] + bbox[2], bbox[1] + bbox[3]
    b4 = bbox[0], bbox[1] + bbox[3]

    cv2.line(kp_mask, b1, b2, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.line(kp_mask, b2, b3, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.line(kp_mask, b3, b4, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.line(kp_mask, b4, b1, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_coco_skeleton(img, kps, kps_lines, alpha=1):
    """
    用于在图像上绘制COCO关键点的骨骼连接线和关节点，并返回带有绘制效果的图像
    功能:
        通过调整 colors 列表中的颜色值，定义了不同部位的颜色。
        复制输入图像 img 到新的图像 kp_mask 中。
        根据给定的关键点坐标和连接关系，绘制关键点之间的连接线和关节点。
        使用 cv2.addWeighted 函数将原始图像 img 与带有绘制效果的 kp_mask 进行混合，以生成最终的图像。
    :param img: 输入图像
    :param kps: 关键点的坐标，形状为 (2, num_kps)，其中 num_kps 是关键点的数量。
    :param kps_lines: 描述骨骼连接关系的索引列表
    :param alpha: 绘制效果的透明度，介于 0 和 1 之间
    :return: 带有绘制效果的图像
    """
    colors = [
        # face
        (255 / 255, 153 / 255, 51 / 255),
        (255 / 255, 153 / 255, 51 / 255),
        (255 / 255, 153 / 255, 51 / 255),
        (255 / 255, 153 / 255, 51 / 255),

        # left arm
        (102 / 255, 255 / 255, 102 / 255),
        (51 / 255, 255 / 255, 51 / 255),

        # right leg
        (255 / 255, 102 / 255, 255 / 255),
        (255 / 255, 51 / 255, 255 / 255),

        # left leg

        (255 / 255, 102 / 255, 102 / 255),
        (255 / 255, 51 / 255, 51 / 255),

        # shoulder-thorax, hip-pevlis,
        (153 / 255, 255 / 255, 153 / 255),  # l shoulder - thorax
        (153 / 255, 204 / 255, 255 / 255),  # r shoulder - thorax
        (255 / 255, 153 / 255, 153 / 255),  # l hip - pelvis
        (255 / 255, 153 / 255, 255 / 255),  # r hip -pelvis

        # center body line
        (255 / 255, 204 / 255, 153 / 255),
        (255 / 255, 178 / 255, 102 / 255),

        # right arm
        (102 / 255, 178 / 255, 255 / 255),
        (51 / 255, 153 / 255, 255 / 255),
    ]

    colors = [[c[2] * 255, c[1] * 255, c[0] * 255] for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    line_thick = 2  # 13
    circle_rad = 2  # 10
    circle_thick = 3  # 7

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if p1 != (0, 0) and p2 != (0, 0):
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=line_thick, lineType=cv2.LINE_AA)
            cv2.circle(
                kp_mask, p1,
                radius=circle_rad, color=colors[l], thickness=circle_thick, lineType=cv2.LINE_AA)
            cv2.circle(
                kp_mask, p2,
                radius=circle_rad, color=colors[l], thickness=circle_thick, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1, kps_scores=None):
    """
    函数用于在图像上绘制关键点和关键点之间的骨骼连接线，还支持显示关键点的置信度分数
    功能:
        使用 plt.get_cmap('rainbow') 创建彩虹色的颜色映射，并将其转换为 OpenCV 颜色格式。
        复制输入图像 img 到新的图像 kp_mask 中。
        根据给定的关键点坐标和连接关系，绘制关键点之间的连接线，并根据置信度阈值过滤关键点。
        如果 kps_scores 不为 None，则在绘制的关键点附近显示置信度分数。
        使用 cv2.addWeighted 函数将原始图像 img 与带有绘制效果的 kp_mask 进行混合，以生成最终的图像。
        效果是在图像上绘制姿态估计的关键点及其骨骼连接线，可以通过调整 kp_thresh 参数来控制关键点置信度的过滤。
    :param img: 输入图像
    :param kps: 关键点的坐标，形状为 (3, num_kps)，其中 num_kps 是关键点的数量。第三 维度表示关键点的置信度分数。
    :param kps_lines: 描述骨骼连接关系的索引列表
    :param kp_thresh: 用于过滤关键点的置信度阈值，低于此阈值的关键点将不会被绘制
    :param alpha: 绘制效果的透明度，介于 0 和 1 之间
    :param kps_scores: 关键点的置信度分数，如果提供，将在关键点周围显示分数
    :return:
    """
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

            if kps_scores is not None:
                cv2.putText(kp_mask, str(kps_scores[i2, 0]), p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_keypoints(img, kps, alpha=1, kps_vis=None):
    """
    函数用于在图像上绘制关键点，支持显示关键点的索引或可选的关键点可见性信息
    功能:
        使用 plt.get_cmap('rainbow') 创建彩虹色的颜色映射，并将其转换为 OpenCV 颜色格式。
        复制输入图像 img 到新的图像 kp_mask 中。
        遍历关键点列表 kps，为每个关键点绘制一个填充的圆，颜色基于索引或彩虹色映射。
        如果提供了 kps_vis，则在绘制的关键点附近显示相应的可见性信息。
        使用 cv2.addWeighted 函数将原始图像 img 与带有绘制效果的 kp_mask 进行混合，以生成最终的图像。
        这个函数的效果是在图像上绘制姿态估计的关键点，并可以显示关键点的可见性信息。可通过调整 alpha 参数来控制绘制效果的透明度。
    :param img: 输入图像
    :param kps: 关键点的坐标列表，每个关键点表示为 (x, y)
    :param alpha: 绘制效果的透明度，介于 0 和 1 之间
    :param kps_vis: 关键点的可见性信息，如果提供，将在关键点附近显示相应的可见性值。
    :return: 图像上绘制姿态估计的关键点，并可以显示关键点的可见性信息的图像
    """
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        if kps_vis is not None:
            cv2.putText(kp_mask, str(kps_vis[i, 0]), p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        else:
            cv2.putText(kp_mask, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_mesh(img, mesh_vertex, alpha=0.5):
    """
    函数用于在图像上绘制三维模型的顶点，每个顶点使用彩虹色进行着色
    功能:
        使用 plt.get_cmap('rainbow') 创建彩虹色的颜色映射，并将其转换为 OpenCV 颜色格式。
        复制输入图像 img 到新的图像 mask 中。
        遍历三维模型的顶点列表 mesh_vertex，为每个顶点绘制一个填充的小圆，颜色基于彩虹色映射。
        使用 cv2.addWeighted 函数将原始图像 img 与带有绘制效果的 mask 进行混合，以生成最终的图像。
        这个函数的效果是在图像上绘制三维模型的顶点，每个顶点用彩虹色进行着色，并可以通过调整 alpha 参数来控制绘制效果的透明度。
    :param img: 输入图像
    :param mesh_vertex: 三维模型的顶点坐标列表，每个顶点表示为 (x, y, z)
    :param alpha: 绘制效果的透明度，介于 0 和 1 之间。
    :return: 在图像上绘制三维模型的顶点的图像
    """
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)


def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):
    """
    函数用于在3D空间中可视化关键点的骨架
    功能:
        创建一个3D子图，使用关键点的3D坐标在3D空间中绘制骨架。
        将每个关键点之间的连接线绘制为连接两个关键点的线段。
        根据 kpt_3d_vis 列表决定是否绘制关键点和连接线。
        如果提供了 filename 参数，则将其用作图像的标题，否则使用默认标题。
        在图像中绘制每个关键点的散点，以及连接线。
        这个函数的效果是在3D空间中可视化关键点的骨架，可以通过提供的参数控制是否绘制关键点和连接线。
    :param kpt_3d: 3D关键点的坐标列表，每个关键点表示为 (x, y, z)
    :param kpt_3d_vis: 关键点是否可见的标志列表
    :param kps_lines: 关键点之间连接的线段列表
    :param filename: 图像标题或文件名（可选）
    :return: 3D空间中可视化关键点的骨架
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])
        # print(z)

        if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
            ax.plot(x, z, -y, color=colors[l], linewidth=2)
        if kpt_3d_vis[i1, 0] > 0:
            ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], color=colors[l], marker='o')
        if kpt_3d_vis[i2, 0] > 0:
            ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], color=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label', labelpad=1)
    ax.set_ylabel('Z Label', labelpad=1)
    ax.set_zlabel('Y Label', labelpad=1)
    # ax.legend()
    ax.set_xticks(np.linspace(-1, 1, 5))  # 设置x轴刻度
    ax.set_yticks(np.linspace(10, 12, 5))  # 设置x轴刻度
    ax.set_xticklabels([''] * 5)
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.show()
    cv2.waitKey(0)


def save_obj(v, f, file_name='output.obj'):
    """
    函数用于将3D模型的顶点和面信息保存到 .obj 文件中
    功能:
        打开指定名称的 .obj 文件进行写操作。
        遍历顶点列表，并将每个顶点的坐标写入 .obj 文件。
        遍历面列表，并将每个面的顶点索引写入 .obj 文件。
        关闭 .obj 文件。
        这个函数的效果是将提供的顶点和面信息保存到 .obj 文件中，以便后续的3D模型渲染和可视化。
    :param v: 顶点坐标的列表，每个顶点表示为 (x, y, z)
    :param f: 面的列表，每个面表示为包含三个或更多顶点索引的整数列表 --> 顶点的索引连接关系
    :param file_name: 输出的 .obj 文件的文件名（可选，默认为 'output.obj'）
    :return: 将提供的顶点和面信息保存到 .obj 文件中
    """
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0] + 1) + '/' + str(f[i][0] + 1) + ' ' + str(f[i][1] + 1) + '/' + str(
            f[i][1] + 1) + ' ' + str(f[i][2] + 1) + '/' + str(f[i][2] + 1) + '\n')
    obj_file.close()


def render_mesh(img, mesh, face, cam_param, color=(1.0, 1.0, 0.9, 1.0)):
    """
    函数用于在渲染的3D模型上叠加一张2D图像
    功能:
        使用提供的3D模型的顶点和面信息创建一个 trimesh 对象。
        将 trimesh 对象转换为 pyrender 中的 Mesh 对象，并设置材质颜色。
        创建一个 pyrender 场景，包括背景颜色、环境光和相机。
        使用提供的相机参数创建一个 pyrender 相机，并将相机添加到场景中。
        创建一个 pyrender 渲染器。
        添加方向光源到场景中。
        使用渲染器对场景进行渲染，获取RGB和深度信息。
        将渲染的RGB图像与输入的2D图像进行叠加，得到最终的结果。
        这个函数的效果是将提供的3D模型渲染到2D图像上，并返回合成后的图像。
    :param img: 输入的2D图像，这是将3D模型渲染上去的背景。
    :param mesh: 3D模型的顶点坐标
    :param face: 3D模型的面信息
    :param cam_param: 3D模型的面信息。
    :param color: 3D模型的颜色，默认为 (1.0, 1.0, 0.9, 1.0)
    :return: 提供的3D模型渲染到2D图像上，并返回合成后的图像
    """
    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=color)
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)
    return img.astype(np.uint8)


def render_mesh_without_image(img, mesh, face, cam_param, color=(1.0, 1.0, 0.9, 1.0)):
    """
    函数用于在渲染的3D模型上叠加一张2D图像
    功能:
        使用提供的3D模型的顶点和面信息创建一个 trimesh 对象。
        将 trimesh 对象转换为 pyrender 中的 Mesh 对象，并设置材质颜色。
        创建一个 pyrender 场景，包括背景颜色、环境光和相机。
        使用提供的相机参数创建一个 pyrender 相机，并将相机添加到场景中。
        创建一个 pyrender 渲染器。
        添加方向光源到场景中。
        使用渲染器对场景进行渲染，获取RGB和深度信息。
        将渲染的RGB图像与输入的2D图像进行叠加，得到最终的结果。
        这个函数的效果是将提供的3D模型渲染到2D图像上，并返回合成后的图像。
    :param img: 输入的2D图像，这是将3D模型渲染上去的背景。
    :param mesh: 3D模型的顶点坐标
    :param face: 3D模型的面信息
    :param cam_param: 3D模型的面信息。
    :param color: 3D模型的颜色，默认为 (1.0, 1.0, 0.9, 1.0)
    :return: 提供的3D模型渲染到2D图像上，并返回合成后的图像
    """
    # mesh
    img_size = img.shape[0:2]
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=color)
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    white_background = np.ones((img_size[0], img_size[1], 3),dtype=np.float32) * 255
    # save to image
    img = rgb * valid_mask + white_background * (1 - valid_mask)
    return img.astype(np.uint8)
