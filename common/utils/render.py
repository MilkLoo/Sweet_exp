import math
import trimesh
import pyrender
import numpy as np
from pyrender.constants import RenderFlags


class WeakPerspectiveCamera(pyrender.Camera):
    """
    代码定义了一个名为 WeakPerspectiveCamera 的类，该类继承自 pyrender.Camera。它代表了一个弱透视相机，用于在三维场景中捕捉图像。
    该类具有以下属性和方法：
        属性：
            scale: 用于控制图像的缩放比例的二维数组，表示在 x 和 y 方向上的缩放因子。
            translation: 用于控制图像的平移的二维数组，表示在 x 和 y 方向上的平移量。
            znear: 相机近裁剪面的距离，默认为 pyrender.camera.DEFAULT_Z_NEAR。
            zfar: 相机远裁剪面的距离，默认为 None。
            name: 相机的名称，默认为 None。
        方法：
            get_projection_matrix(width=None, height=None): 返回相机的投影矩阵。参数 width 和 height 可以指定图像的宽度和高度。
    在方法 get_projection_matrix 中，根据 scale 和 translation 属性，构造了一个投影矩阵 P，用于将三维空间中的点投影到相机的图像平面上。
    """

    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:
    """
    代码定义了一个名为 Renderer 的类，用于渲染三维模型并将其投影到图像上。以下是该类的主要属性和方法：
    属性：
        resolution: 一个包含两个整数的元组，表示渲染图像的分辨率。
        faces: 三角形面的索引，用于定义三角形网格。
        orig_img: 一个布尔值，指示是否在图像上叠加原始图像。
        wireframe: 一个布尔值，指示是否以线框模式渲染模型。
        renderer: pyrender.OffscreenRenderer 的实例，用于执行渲染操作。
        scene: pyrender.Scene 的实例，表示渲染场景。
    方法：
        render(img, verts, cam, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9], rotate=False):执行渲染操作。
        参数包括：
            img: 原始图像。
            verts: 三维模型的顶点坐标。
            cam: 包含四个参数的元组，表示相机的缩放因子和平移量。
            angle 和 axis: 可选参数，表示旋转角度和旋转轴。
            mesh_filename: 可选参数，表示将模型导出到文件的路径。
            color: 可选参数，表示模型的颜色。
            rotate: 一个布尔值，指示是否对模型进行旋转。
    该类的主要功能是将给定的三维模型渲染到图像上，并根据相机参数将其投影到图像平面上。
    """
    def __init__(self, face, resolution=(224, 224), orig_img=False, wireframe=False):
        self.resolution = resolution

        self.faces = face
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        # light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

    def render(self, img, verts, cam, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9], rotate=False):

        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        if rotate:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(60), [0, 1, 0])
            mesh.apply_transform(rot)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            smooth=True,
            wireframe=True,
            roughnessFactor=1.0,
            emissiveFactor=(0.1, 0.1, 0.1),
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.2,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, depth = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (depth > 0)[:, :, np.newaxis]
        output_img = rgb * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image
