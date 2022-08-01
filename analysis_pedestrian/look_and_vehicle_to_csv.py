# 分析行人看与未看情况下车辆的行为
# 'look': {'not-looking': 0, 'looking': 1},
# 'vehicle': {'stopped': 0, 'moving_slow': 1, 'moving_fast': 2,
#             'decelerating': 3, 'accelerating': 4},
# [11153, 1429, 1980, 30078, 18072]
# <attribute name="id">0_1_3b</attribute>
# <attribute name="old_id">pedestrian1</attribute>
# <attribute name="look">not-looking</attribute>
# <attribute name="reaction">clear_path</attribute>
# <attribute name="action">walking</attribute>
# <attribute name="cross">not-crossing</attribute>
# <attribute name="hand_gesture">__undefined__</attribute>
# <attribute name="occlusion">full</attribute>
# <attribute name="nod">__undefined__</attribute>
# 每一帧车辆状态、最近的五个行人look、not-look及box[宽、高]（按box高度区分远近）、
import numpy as np

import config
from toolkit.alpha_pose_to_csv import box_iou, get_key_points, np_sort
from toolkit.plot_data import plot_pose_box_look
from toolkit.read_pose_data import read_json
from toolkit.tool import xml_read, str_to_int


def get_train_data(jaad_anno_path, jaad_vehicle_path, int_video_id, uuid):
    vehicle_anno = xml_read(jaad_vehicle_path)
    vehicle_info = vehicle_anno["vehicle_info"]["frame"]
    vehicle_action_list = []
    frame_all_id = 0
    cross_label_bool = False
    for v_info in vehicle_info:
        action = v_info["@action"]
        vehicle_action_dict = {'stopped': 0, 'moving_slow': 1, 'moving_fast': 2, 'decelerating': 3, 'accelerating': 4}
        action_num = vehicle_action_dict[action]
        vehicle_id = int(v_info["@id"])
        if frame_all_id != vehicle_id:
            print("error, vehicle id is not equal check id.", vehicle_id, frame_all_id)
            break
        frame_all_id += 1
        vehicle_action_list.append(action_num)
    output = np.zeros((frame_all_id, 3 + 6))
    jaad_anno = xml_read(jaad_anno_path)
    annotations = jaad_anno["annotations"]
    if "track" not in annotations:
        return np.mat([]),False
    if "box" in annotations["track"]:
        track_box = [annotations["track"]]
    else:
        track_box = annotations["track"]
    is_repeat = 0
    for i in track_box:
        if i["@label"] != "pedestrian":
            continue
        is_repeat += 1
        if is_repeat >= 2:
            print(jaad_anno_path, "there are two box in one data_by_video")
        for annotation in i["box"]:
            # jaad 注释文件，左上角，右下角 (top-left, bottom-right), ybr>xbr,ytl>xtl
            img_frame_id = int(annotation["@frame"])
            xtl, ytl = str_to_int(annotation["@xtl"]), str_to_int(annotation["@ytl"])
            xbr, ybr = str_to_int(annotation["@xbr"]), str_to_int(annotation["@ybr"])
            x_mid, y_mid = (xtl + xbr) // 2, (ytl + ybr) // 2
            width, height = xbr - xtl, ybr - ytl
            is_look = annotation["attribute"][2]["#text"]
            cross = annotation["attribute"][5]["#text"]
            look_label = 0
            if is_look == "looking":
                look_label = 1
            cross_label = 0
            if cross == "crossing":
                cross_label = 1
                cross_label_bool = True
            if height > output[img_frame_id][-2]:
                output[img_frame_id] = np.mat(
                    [int_video_id, img_frame_id, vehicle_action_list[img_frame_id], x_mid, y_mid, width,
                     height, look_label, cross_label])
    # video_pose_box.release()
    print(int_video_id, "shape:", output.shape)
    return output, cross_label_bool


def get_init_data():
    video_count = 0  # 计算有多少个视频是有效的
    xml_anno = config.jaad_anno
    xml_vehicle_anno = config.jaad_vehicle
    uuid = 0
    cross_list = []  # 保存有cross的视频列表，如果某视频完全没有cross过程，则不保存
    for i in range(1, 347):
        video_id_name = "video_" + str(i).zfill(4)
        xml_anno_path = xml_anno + video_id_name + ".xml"
        xml_vehicle_path = xml_vehicle_anno + video_id_name + "_vehicle.xml"
        x, cross_bool = get_train_data(xml_anno_path, xml_vehicle_path, i, uuid)
        if cross_bool:
            cross_list.append(i)
        print("x.shape", x.shape[0], x.shape[1])
        if x.shape[1] > 1:
            uuid += x.shape[0]
            video_count += 1
            np.savetxt("./data/data" + str(i) + ".csv", x, delimiter=',')
    print(cross_list)
    return video_count


if __name__ == "__main__":
    get_init_data()
