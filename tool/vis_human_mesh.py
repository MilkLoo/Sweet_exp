"""
   通过导入 .obj文件 进行可视化 也可以通过 blender 软件进行显示。
"""

import meshio
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 读取 .obj 文件   这里修改文件路径
    mesh = meshio.read("/home/ly/yxc_exp_smpl/demo/output/012_0.obj")

    # 提取顶点和面
    vertices = mesh.points
    faces = mesh.cells[0].data
    # 创建一个 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制模型
    for face in faces:
        verts = [vertices[i] for i in face]
        verts.append(verts[0])  # 闭合面
        xs, ys, zs = zip(*verts)
        tri = ax.plot_trisurf(xs, ys, zs, triangles=[list(range(len(verts)))], color=(1.0, 0.75, 0.8, 1.0))
        # tri.set_facecolor((0.5, 0.5, 0.5, 1.0))  # 设置三角面片的颜色
        # ax.plot(xs, ys, zs)

    # 设置图形参数
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('OBJ Model')

    # 隐藏刻度
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # 隐藏轴和刻度
    ax.axis('off')

    # 显示图形
    plt.show()
