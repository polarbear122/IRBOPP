"""
读取alpha pose的结果，在训练阶段，将json数据转换成csv数据，存储一包含video_id，idx，img_id，keypoints，box，label的向量。
以video_id，idx，img_id的顺序排序
"""

import csv
import os.path

import numpy as np
# 提供读取数据的方法
# 从alpha pose的检测结果和jaad的注释文件中读取keypoints和对应img id，保存结果为csv文件
import pandas as pd

from JAAD_2.jaad_data import JAAD
from toolkit.read_pose_data import read_json
from toolkit.tool import get_anno_by_frame_id, get_anno_by_list, change_str_id_to_int


def get_key_points(keypoints: list):
    # key points [0, 1, 2, 3, 4, 17, 18]
    if len(keypoints) != 26 * 3 and len(keypoints) != 17 * 3:
        print('the len of keypoints is not 26*3 or 17*3')
        return []
    key_points = []
    # for i in [0, 1, 2, 3, 4, 17, 18]:
    #     key_points.append(keypoints[i * 3 + 2])
    for i in range(len(keypoints) // 3):
        key_points.append(keypoints[i * 3 + 0])
        key_points.append(keypoints[i * 3 + 1])
        key_points.append(keypoints[i * 3 + 2])
    return key_points


def box_iou(box1, box2):
    # 计算box a和box b的IOU值,输入左上角坐标，右下角坐标, box:[x1, y1, x2, y2]
    # 例如box1 = [0,0,10,10], box2 = [5,5,15,15]
    in_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    in_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inner = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inner
    if union == 0.0:
        return 0.0
    iou = inner / union
    return iou


def gen_result_label(each_pedestrian_anno, uuid, i):
    need_attribute = ['age', 'gender', 'motion_direction', 'crossing']
    attributes_label = get_anno_by_list(each_pedestrian_anno['attributes'], need_attribute)
    if attributes_label[-1] == -1:
        attributes_label[-1] = 0
    need_behavior = ['look', 'cross', 'reaction', 'hand_gesture', 'action']
    behavior_label = get_anno_by_frame_id(each_pedestrian_anno['behavior'], need_behavior, i)
    need_appearance = ['pose_front', 'pose_back', 'pose_left', 'pose_right']
    appearance_label = get_anno_by_frame_id(each_pedestrian_anno['appearance'], need_appearance, i)
    result_l = [uuid] + attributes_label + behavior_label + appearance_label
    return result_l


def get_train_data(alpha_pose_path, video_id_name, int_video_id, uuid):
    x = []
    alpha_pose = read_json(alpha_pose_path)
    id_uuid_list = []
    this_video_anno = jaad_data[video_id_name]
    ped_anno = this_video_anno['ped_annotations']
    for each_pedestrian_id in ped_anno:
        ped_id = change_str_id_to_int(each_pedestrian_id)
        each_pedestrian_anno = ped_anno[each_pedestrian_id]
        if not each_pedestrian_anno['appearance'] or not each_pedestrian_anno['attributes']:
            continue
        for i in range(len(each_pedestrian_anno['frames'])):
            # frame_id:此行人出现在第几帧中，与i不同。i是指在这个行人的列表中，该帧是第几个
            frame_id = each_pedestrian_anno['frames'][i]
            # print(i, each_pedestrian_id, frame_id)
            true_box = each_pedestrian_anno['bbox'][i]
            # jaad 注释文件,左上角，右下角 (top-left,bottom-right), ytl>xtl, ybr>xbr
            result_l = gen_result_label(each_pedestrian_anno, uuid, i)

            max_iou = max_iou_threshold = 0.6
            pose_feature_idx = 0
            # alpha pose的box位置,格式为([0],[1])左上角,([2],[3])宽和高,修改成(左上角,右下角)格式
            x_keypoints_proposal, max_pose_box = [], []  # 存储key points,max_pose_box为iou最大时的box（左上角，宽高）格式
            for j in range(len(alpha_pose)):
                pose = alpha_pose[j]
                if pose['score'] < 1:
                    continue
                if pose['image_id'] == str(frame_id) + '.jpg':
                    pose_box = [pose['box'][0], pose['box'][1], pose['box'][0] + pose['box'][2],
                                pose['box'][1] + pose['box'][3]]
                    tl_width_height_box = pose['box']  # 获取alpha pose中的box，(左上角点，宽高)格式
                    iou_val = box_iou(pose_box, true_box)
                    if iou_val > max_iou:
                        x_keypoints_proposal = get_key_points(pose['keypoints'])
                        max_pose_box = tl_width_height_box
                        max_iou = iou_val
                        pose_feature_idx = j
                elif pose['image_id'] == str(frame_id + 1) + '.jpg':
                    break
            pose_in_img = max_pose_box and 0 < max_pose_box[0] < 1920 and 0 < max_pose_box[1] < 1080 and 0 < \
                          max_pose_box[2] < 1920 and 0 < max_pose_box[3] < 1080
            if x_keypoints_proposal and max_iou > max_iou_threshold and pose_in_img:
                x.append([uuid, int_video_id, frame_id, ped_id] + x_keypoints_proposal + max_pose_box + [
                    pose_feature_idx])
                id_uuid_list.append(result_l)
                uuid += 1
    print(video_id_name, 'shape:', np.mat(x).shape)
    return np.mat(x), id_uuid_list


# 传入一个numpy数组，按第一列、第二列、第三列的顺序对numpy数组进行排序
def np_sort(n_arr: np.array):
    _a = n_arr
    _a = _a[np.lexsort((_a[:, 2], _a[:, 1], _a[:, 0]))]
    return _a


def get_init_data():
    alpha_pose = 'res/'
    uuid = 0
    id_uuid_list_all = []
    all_result = None
    for i in range(1, 347):
        video_id_name = 'video_' + str(i).zfill(4)
        alpha_pose_path = alpha_pose + video_id_name + '/alphapose-results.json'
        if not os.path.exists(alpha_pose_path):
            continue
        x, id_uuid_list = get_train_data(alpha_pose_path, video_id_name, i, uuid)
        id_uuid_list_all.append(id_uuid_list)

        if x.shape[1] > 1:
            uuid += x.shape[0]
            if all_result is None:
                all_result = x
            else:
                all_result = np.concatenate((all_result, x), axis=0)
        print('all_result.shape', all_result.shape[0], all_result.shape[1])
    np.savetxt('save_data/new_pose.csv', all_result, delimiter=',', fmt='%.3e')
    with open('save_data/new_label.csv', 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        for i in range(len(id_uuid_list_all)):
            for j in range(len(id_uuid_list_all[i])):
                row = id_uuid_list_all[i][j]
                csv_writer.writerow(row)


def rename_uuid():
    pose_arr = pd.read_csv('train/pose_all.csv', header=None, sep=',', encoding='utf-8').values
    for i in range(len(pose_arr)):
        pose_arr[i, 0] = i
    np.savetxt('train/pose_all_re.csv', pose_arr, delimiter=',')


if __name__ == '__main__':
    x = JAAD(data_path='JAAD_2/')
    jaad_data = x.generate_database()
    get_init_data()
