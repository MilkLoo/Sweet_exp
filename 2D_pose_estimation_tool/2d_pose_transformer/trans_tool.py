"""
   需要将2D pose 的格式转换为我的项目中的demo的2D pose result的形式，以及test的格式。

   demo的格式
   {"image_name": [人数[ 关键点个数 [坐标信息]]]}
"""
import copy


# 单张
def trans_my_project_format_list(object_dict):
    my_project_format_list_1 = []
    my_project_format_list_2 = []
    for key in object_dict.keys():
        for key_1 in sorted(object_dict[key].keys()):
            my_project_format_list_2.append(object_dict[key][key_1])
        deep_copy_data = copy.deepcopy(my_project_format_list_2)
        my_project_format_list_1.append(deep_copy_data)
        my_project_format_list_2.clear()
    return my_project_format_list_1


def trans_my_project_format_list_batch(object_dict):
    my_project_format_list_2 = []
    for key in object_dict.keys():
        for key_1 in sorted(object_dict[key].keys()):
            my_project_format_list_2.append(object_dict[key][key_1])
    return my_project_format_list_2


# 这个适合单张图片
def trans_my_demo_format_dict(object_list, image_name_str):
    my_project_format_dict = {image_name_str: object_list}
    return my_project_format_dict
