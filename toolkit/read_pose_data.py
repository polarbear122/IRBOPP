# 读取alpha pose的检测结果 json

import json
import os.path
import random

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def read_json(json_path):
    json_data = open(json_path)
    json_string = json_data.read()
    j = json.loads(json_string)
    return j


# From https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def box_IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(boxAArea + boxBArea - inter_area)

    # return the intersection over union value
    return iou


def read_bounding_box(txtFilePath):
    file1 = open(txtFilePath, "r")
    bbox = file1.readlines()[0].split(',')[1:5]
    file1.close()
    bbox_x = int(bbox[0])
    bbox_y = int(bbox[1])
    bbox_w = int(bbox[2])
    bbox_h = int(bbox[3])
    bbox = [bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h]
    return bbox


def read_pose_box(j_image):
    # x, y are the vectors of body parts locations, confi is the vector of the confidence
    x, y, confi = [], [], []  # x位置 # y位置 # 特征点的可见性
    # 提取出头部的几个特征点  0 1 2 3 4 17 18
    right_x = 0
    left_x = 99999
    top_y = 0
    bottom_y = 99999
    for i in range(26):
        x.append(j_image["keypoints"][i * 3 + 0])
        y.append(j_image["keypoints"][i * 3 + 1])
        confi.append(j_image["keypoints"][i * 3 + 2])

        # 获取特征点位置，避免越界
        if x[-1] > right_x:
            right_x = x[-1]
        if x[-1] < left_x:
            left_x = x[-1]
        if y[-1] > top_y:
            top_y = y[-1]
        if y[-1] < bottom_y:
            bottom_y = y[-1]
        # Visualize Json result
        # plt.scatter(x, y, s=2, c='red', marker='o')

    # 获取 bounding box
    pbox = [0, 0, 0, 0]
    pbox[0], pbox[1], pbox[2], pbox[3] = left_x, bottom_y, right_x, top_y
    return x, y, confi, pbox


def write_pose_cross_data(j, bbox_path, image_path, output_data_path, tag, vis=False):
    result = {}
    count = 0

    for j_image in j:
        # Enough confidence for pedestrian
        if j_image['score'] > 1.2:
            image_ID = j_image['image_id']
            tokens = image_ID.split('_')
            vid = tokens[0]
            vnum = tokens[1]
            pid = tokens[2]
            pnum = 'B' + tokens[3][1:-4]
            txtFileName = vid + '_' + vnum + '_' + pid + '_' + pnum + '.txt'
            txtFilePath = bbox_path + '/' + txtFileName

            bbox = read_bounding_box(txtFilePath)
            x, y, confi, pbox = read_pose_box(j_image)

            if vis:
                # Show original image_patch at 1920 * 1080
                img = mpimg.imread(image_path + '/' + image_ID)
                fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
                ax = fig.gca()
                ax.imshow(img)
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1,
                                         edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                rect = patches.Rectangle((pbox[0], pbox[1]), pbox[2] - pbox[0], pbox[3] - pbox[1], linewidth=1,
                                         edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                plt.show()
                plt.close()

            iou = box_IoU(pbox, bbox)

            # If iou is not 0
            if iou > 0.0000000001:
                # write txt file
                strout = ''
                for i in range(26):
                    strout += str(x[i]) + ',' + str(y[i]) + ',' + str(confi[i]) + ','
                strout += str(iou) + ',' + tag
                str0 = result.get(image_ID)
                if str0 is not None:
                    oldiou = float(str0.split(',')[-2])
                    if oldiou < iou:
                        result[image_ID] = strout
                else:
                    result[image_ID] = strout
        count += 1
        print(count)
    test = {}
    train = {}
    for key, value in result.items():
        if random.random() < 0.2:
            test[key] = value
        else:
            train[key] = value

    outFilePath = output_data_path + '/' + tag + '_all_data.json'
    outTestFilePath = output_data_path + '/' + tag + '_test_data.json'
    outTrainFilePath = output_data_path + '/' + tag + '_train_data.json'
    with open(outFilePath, 'w') as outfile:
        json.dump(result, outfile)
    with open(outTestFilePath, 'w') as outfile:
        json.dump(test, outfile)
    with open(outTrainFilePath, 'w') as outfile:
        json.dump(train, outfile)


def main():
    json_path = "E:/CodeResp/pycode/DataSet/pose_result/alphapose-results-0002.json"
    # bbox_path = '../../Dataset/Data_by_Matlab/cross/bbox'
    # image_path = '../../Dataset/Data_by_Matlab/cross/image_patch'
    # output_data_path = '../../Dataset/Data_by_Matlab/cross_pose_data'
    #
    # j = readJson(json_path)
    # write_pose_cross_data(j, bbox_path, image_path, output_data_path, 'cross', True)


def compare_json():
    x = read_json('C:/Users/60260/Desktop/1.json')
    y = read_json('C:/Users/60260/Desktop/2.json')
    print(len(x), len(y))
    pass


def compare_json_len():
    for i in range(347):
        alpha_pose_path = 'res/video_' + str(i).zfill(4) + '/alphapose-results.json'
        if not os.path.exists(alpha_pose_path):
            print(alpha_pose_path)
            continue
        alpha_pose = read_json(alpha_pose_path)
        print(i, len(alpha_pose))


if __name__ == "__main__":
    compare_json_len()
