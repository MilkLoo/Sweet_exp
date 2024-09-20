import cv2
import argparse
from config import cfg
import torch.backends.cudnn as cudnn
import ffmpeg
import os.path as osp
from torch.nn.parallel.data_parallel import DataParallel
from model import get_mesh_model
# from model_transformer import get_mesh_model
import torch
from common.utils.smpl import SMPL
from libary.network.rtpose_vgg import get_model
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from libary.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans, human_coord, \
    get_coco_coord
from libary.utils.paf_to_pose import paf_to_pose_cpp
from libary.config import cfg, update_config
import numpy as np
import copy
import torchvision.transforms as transforms
from config import config
from common.utils.preprocessing import process_bbox, generate_patch_image, get_bbox
from common.utils.transforms import pixel2cam, cam2pixel, transform_joint_to_other_db
from common.utils.vis import vis_mesh, save_obj, render_mesh, vis_coco_skeleton, render_mesh_without_image
import colorsys


def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if 'stream' in meta_dict and "tags" in meta_dict['streams'][0] and 'rotate' in meta_dict['streams'][0]['tags']:
        rotation = int(meta_dict['streams'][0]['tags']['rotate'])
        if rotation == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        elif rotation == 180:
            rotateCode = cv2.ROTATE_180
        elif rotation == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotateCode


def correct_rotation(frame, rotateCode):
    return cv2.rotate(frame, rotateCode)


def trans_my_project_format_list_batch(object_dict):
    my_project_format_list_2 = []
    for key in object_dict.keys():
        for key_1 in sorted(object_dict[key].keys()):
            my_project_format_list_2.append(object_dict[key][key_1])
    return my_project_format_list_2


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


def add_pelvis(joint_coord, joints_name):
    lhip_idx = joints_name.index('L_Hip')
    rhip_idx = joints_name.index('R_Hip')
    pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
    pelvis[2] = joint_coord[lhip_idx, 2] * joint_coord[rhip_idx, 2]  # confidence for openpose
    pelvis = pelvis.reshape(1, 3)

    joint_coord = np.concatenate((joint_coord, pelvis))

    return joint_coord


def add_neck(joint_coord, joints_name):
    lshoulder_idx = joints_name.index('L_Shoulder')
    rshoulder_idx = joints_name.index('R_Shoulder')
    neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
    neck[2] = joint_coord[lshoulder_idx, 2] * joint_coord[rshoulder_idx, 2]
    neck = neck.reshape(1, 3)

    joint_coord = np.concatenate((joint_coord, neck))

    return joint_coord


joint_num = 30  # original: 24. manually add nose, L/R eye, L/R ear, head top
joints_name = (
    'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe',
    'R_Toe',
    'Neck', 'L_Thorax', 'R_Thorax',
    'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose',
    'L_Eye',
    'R_Eye', 'L_Ear', 'R_Ear', 'Head_top')
flip_pairs = (
    (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))
skeleton = (
    (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
    (17, 19),
    (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 24), (24, 15), (24, 25),
    (24, 26),
    (25, 27), (26, 28), (24, 29))

# SMPl mesh
vertex_num = 6890
smpl = SMPL()
face = smpl.face
joint_regressor = smpl.joint_regressor

# other joint set
coco_joints_name = (
    'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
    'R_Wrist',
    'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
coco_skeleton = (
    (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15),
    (5, 6),
    (11, 17), (12, 17), (17, 18))

vis_joints_name = (
    'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
    'R_Wrist',
    'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Thorax', 'Pelvis')
vis_skeleton = (
    (0, 1), (0, 2), (2, 4), (1, 3), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 17), (6, 17),
    (11, 18),
    (12, 18), (17, 18), (17, 0), (6, 8), (8, 10),)

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='/home/ly/yxc_exp_smpl/2D_pose_estimation_tool/experiments/vgg19_368x368_sgd.yaml',
                    type=str)
parser.add_argument('--weight', type=str,
                    default='/home/ly/yxc_exp_smpl/2D_pose_estimation_tool/model_weight/pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
parser.add_argument('--gpu', default="0", type=str, dest='gpu_ids')
# 这里改模型权重文件，不同的实验模型不一样。做完实验之后会在 output 文件下的 README.md 文件进行说明。
parser.add_argument('--model_path', type=str,
                    default='/home/ly/yxc_exp_smpl/output/exp_3/exp_04-29_18:43/checkpoint/snapshot_14.pth.tar')
# parser.add_argument('--model_path', type=str,
#                     default='/home/ly/yxc_exp_smpl/output/exp_4/exp_05-22_11:20/checkpoint/snapshot_22.pth.tar')

args = parser.parse_args()

# test gpus
if not args.gpu_ids:
    assert 0, print("Please set proper gpu ids")

if '-' in args.gpu_ids:
    gpus = args.gpu_ids.split('-')
    gpus[0] = int(gpus[0])
    gpus[1] = int(gpus[1]) + 1
    args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

# update config file
update_config(cfg, args)

model = get_model('vgg19')
model.load_state_dict(torch.load(args.weight))
model.cuda()
model.float()
model.eval()

if __name__ == "__main__":
    # 视频处理
    # video_path = '/home/ly/yxc_exp_smpl/demo_video/input/basketball.mp4'
    # video_path = '/home/ly/yxc_exp_smpl/demo_video/input/basketball_1.mp4'
    # video_path = '/home/ly/yxc_exp_smpl/demo_video/input/run.mp4'
    # video_path = '/home/ly/yxc_exp_smpl/demo_video/input/dance.mp4'
    # video_path = '/home/ly/yxc_exp_smpl/demo_video/input/crossban.mp4'
    # video_path = '/home/ly/yxc_exp_smpl/demo_video/input/clame.mp4'
    # video_path = '/home/ly/yxc_exp_smpl/demo_video/input/clamp.mp4'
    # video_path = '/home/ly/yxc_exp_smpl/demo_video/input/clamp_1.mp4'
    # video_path = '/home/ly/yxc_exp_smpl/demo_video/input/walk.mp4'
    # video_path = "/home/ly/yxc_exp_smpl/2D_lightweight_human_pose_estimation/dance.mp4"
    video_path = "/home/ly/yxc_exp_smpl/demo_video/input/gripper.mp4"
    video_capture_dummy = cv2.VideoCapture(video_path)
    if video_capture_dummy is None:
        assert FileExistsError
    ret, oriImg = video_capture_dummy.read()
    shape_tuple = tuple(oriImg.shape[1::-1])
    print("Shape of image is ", shape_tuple)
    rotate_code = check_rotation(video_path)
    video_capture_dummy.release()

    video_capture = cv2.VideoCapture(video_path)

    oriImg_list = []
    keypoints = []
    mesh_list = []
    all_mesh = []
    bbox_list = []
    all_bbox = []
    princpt_list = []
    all_princpt = []

    while video_capture.isOpened():
        # Capture frame-by-frame
        try:
            ret, oriImg = video_capture.read()
            if rotate_code is not None:
                oriImg = correct_rotation(oriImg, rotate_code)
            oriImg_list.append(oriImg)
            cv2.imshow('Video', oriImg)
            # vid_out.write(out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            break
    video_capture.release()
    cv2.destroyAllWindows()

    print("Number of frames", len(oriImg_list))

    # 处理2D姿态
    count = 0
    for oriImg in oriImg_list:
        count += 1
        if count == len(oriImg_list):
            print(count, "frames processed")
            print("=" * 20)

        try:
            shape_dst = np.min(oriImg.shape[0:2])
        except:
            break
        with torch.no_grad():
            paf, heatmap, imscale = get_outputs(
                oriImg, model, 'rtpose')

        humans = paf_to_pose_cpp(heatmap, paf, cfg)
        human_coord_info, human_num = human_coord(oriImg, humans)
        all_human_coco_coord = get_coco_coord(human_coord_info, human_num)
        my_format = trans_my_project_format_list(all_human_coco_coord)
        keypoints.append(my_format)

    if keypoints is not None:
        print("2D pose Successful loading ... ")

    print("Load Mesh Model ...")
    # 设置网格生成模型
    config.set_args(args.gpu_ids, is_test=True)
    cudnn.benchmark = True
    config.render = True

    # snapshot load  加载好模型
    model_path = args.model_path
    assert osp.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    model = get_mesh_model(vertex_num, joint_num, 'test')

    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['network'], strict=False)
    model.eval()

    # 利用图像和2D姿态生成人体网格
    transform = transforms.ToTensor()

    for frame, keypoint in zip(oriImg_list, keypoints):
        inputs2 = frame.copy()
        original_img_height, original_img_width = frame.shape[:2]
        mesh_list.clear()
        bbox_list.clear()
        princpt_list.clear()
        for idx in range(len(keypoint)):
            """ 2D pose input setting & hard-coding for filtering """
            pose_thr = 0.1
            coco_joint_img = np.asarray(keypoint[idx])[:, :3]
            coco_joint_img = add_pelvis(coco_joint_img, coco_joints_name)
            coco_joint_img = add_neck(coco_joint_img, coco_joints_name)
            coco_joint_valid = (coco_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32)

            det_score = sum(coco_joint_img[:, 2])
            if det_score < 1.0:
                continue
            if len(coco_joint_img[:, 2:].nonzero()[0]) < 1:
                continue
            # filter the same targets
            tmp_joint_img = coco_joint_img.copy()
            continue_check = False

            """ Prepare model input """
            # prepare bbox
            bbox = get_bbox(coco_joint_img, coco_joint_valid[:, 0])  # xmin, ymin, width, height
            bbox = process_bbox(bbox, original_img_width, original_img_height)
            if bbox is None:
                continue
            img, img2bb_trans, bb2img_trans = generate_patch_image(inputs2[:, :, ::-1], bbox, 1.0, 0.0, False,
                                                                   config.input_img_shape)
            img = transform(img.astype(np.float32)) / 255
            img = img.cuda()[None, :, :, :]

            coco_joint_img_xy1 = np.concatenate((coco_joint_img[:, :2], np.ones_like(coco_joint_img[:, :1])), 1)
            coco_joint_img[:, :2] = np.dot(img2bb_trans, coco_joint_img_xy1.transpose(1, 0)).transpose(1, 0)
            coco_joint_img[:, 0] = coco_joint_img[:, 0] / config.input_img_shape[1] * config.output_hm_shape[2]
            coco_joint_img[:, 1] = coco_joint_img[:, 1] / config.input_img_shape[0] * config.output_hm_shape[1]

            coco_joint_img = transform_joint_to_other_db(coco_joint_img, coco_joints_name, joints_name)
            coco_joint_valid = transform_joint_to_other_db(coco_joint_valid, coco_joints_name, joints_name)
            coco_joint_valid[coco_joint_img[:, 2] <= pose_thr] = 0

            # check truncation
            coco_joint_trunc = coco_joint_valid * (
                    (coco_joint_img[:, 0] >= 0) * (coco_joint_img[:, 0] < config.output_hm_shape[2]) * (
                    coco_joint_img[:, 1] >= 0) * (coco_joint_img[:, 1] < config.output_hm_shape[1])).reshape(
                -1, 1).astype(np.float32)
            coco_joint_img, coco_joint_trunc, bbox = torch.from_numpy(coco_joint_img).cuda()[None, :,
                                                     :], torch.from_numpy(
                coco_joint_trunc).cuda()[None, :, :], torch.from_numpy(bbox).cuda()[None, :]

            """ Model forward """
            inputs = {'img': img, 'joints': coco_joint_img, 'joints_mask': coco_joint_trunc}
            targets = {}
            meta_info = {'bbox': bbox}
            with torch.no_grad():
                out = model(inputs, targets, meta_info, 'test')
            mesh_cam_render = out['mesh_cam_render'][0].cpu().numpy()
            mesh_list.append(mesh_cam_render)
            bbox = out['bbox'][0].cpu().numpy()
            bbox_list.append(bbox)
            princpt = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
            princpt_list.append(princpt)
        all_mesh.append(mesh_list.copy())
        all_bbox.append(bbox_list.copy())
        all_princpt.append(princpt_list.copy())

    if all_mesh is not None:
        print("Mesh successful loading ...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_out = cv2.VideoWriter('/home/ly/yxc_exp_smpl/demo_video/output/gripper.mp4', fourcc, 20.0, shape_tuple)

    image_list = []

    color = colorsys.hsv_to_rgb(0.2, 0.5, 1.0)
    for frame, mesh, princpt in zip(oriImg_list, all_mesh, all_princpt):
        # frame_size = frame.shape[:2]
        frame_copy = frame.copy()
        for idx in range(len(mesh)):
            # frame_copy = render_mesh(frame_copy, mesh[idx], face, {'focal': (10000,10000), 'princpt': princpt[idx]},
            #                          color=color)

            frame_copy = render_mesh_without_image(frame_copy, mesh[idx], face,
                                                   {'focal': config.focal, 'princpt': princpt[idx]},
                                                   color=color)

        image_list.append(frame_copy)

    for i in image_list:
        vid_out.write(i)

    print("Video successful loading !!!")
