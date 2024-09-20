import cv2
import os
import json
from PIL import Image


def video_to_frames(video_path, output_folder, json_file):
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否打开成功
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    frame_count = 0
    image_dict = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每一帧保存为图片
        frame_name = f"image_{frame_count:04d}.jpg"
        frame_path = os.path.join(output_folder, frame_name)

        # 使用PIL保存图片
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.save(frame_path)

        # 将图片名称作为键，图片路径作为值保存到字典中
        image_dict[frame_name] = frame_path
        frame_count += 1

    # 释放视频对象
    cap.release()

    # 将字典保存到JSON文件中
    with open(json_file, 'w') as json_f:
        json.dump(image_dict, json_f, indent=4)

    print(f"Processed {frame_count} frames and saved to {json_file}.")


# 使用示例
video_path = "/home/ly/yxc_exp_smpl/demo_video/input/walk_a.mp4"  # 替换为你的视频文件路径
output_folder = "/home/ly/yxc_exp_smpl/image_output/walk_a"  # 替换为保存帧的输出文件夹
json_file = "/home/ly/yxc_exp_smpl/image_output/Json/walk_a.json"  # 替换为输出的json文件名称

video_to_frames(video_path, output_folder, json_file)
