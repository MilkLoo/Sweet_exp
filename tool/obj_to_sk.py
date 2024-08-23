"""
这个是根据.obj文件 得到 3D 骨架坐标点信息
"""
import numpy as np
from common.utils.smpl import SMPL
import json


def read_obj(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                # 提取顶点坐标数据并将其转换为浮点数列表
                vertex = [float(x) for x in line.strip().split()[1:]]
                vertices.append(vertex)
    return vertices


smpl = SMPL()
joint_regressor = smpl.joint_regressor
joint_3d_dict = {}
if __name__ == "__main__":
    # 示例：读取OBJ文件中的顶点坐标数据
    obj_file = '/home/ly/yxc_exp_smpl/demo/output/011._0.obj'
    vertex_data = np.array(read_obj(obj_file))
    joint_3d = np.dot(joint_regressor, vertex_data).tolist()
    joint_3d_dict[f"people_0"] = joint_3d
    file_path = "/home/ly/yxc_exp_smpl/tool/3d_keypoint_data/obj_to_sk.json"
    with open(file_path, "w") as json_files:
        json.dump(joint_3d_dict, json_files)
    print("JSON 文件保存成功:", file_path)
