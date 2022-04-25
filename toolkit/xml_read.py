# 仅能读取小文件，注释超过300的文件会读取出错
import logging

import cv2
import xmltodict


def xml_read(xml_file_path):
    with open(xml_file_path) as xml_file:
        # xmltodict.parse()方法可以将xml数据转为python中的dict字典数据
        parser_data = xmltodict.parse(xml_file.read())
    xml_file.close()
    return parser_data


# 读取JAAD的xml注释文件
def jaad_read(xml_file_path, output_file, video_id):
    parser_data = xml_read(xml_file_path)
    annotations = parser_data["annotations"]
    if "track" not in annotations:
        logging.info(xml_file_path, "no track exist")
        print(xml_file_path, "no track exist")
        return
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
            logging.info(xml_file_path, "there are two box in one video")
            print(xml_file_path, "there are two box in one video")
        for annotation in i["box"]:
            is_look = annotation["attribute"][2]["#text"]
            xbr = str_to_int(annotation["@xbr"])  # 左上角，右下角 (top-left, bottom-right)
            xtl = str_to_int(annotation["@xtl"])
            ybr = str_to_int(annotation["@ybr"])  # ybr>xbr,ytl>xtl
            ytl = str_to_int(annotation["@ytl"])
            x_mid, y_mid = (xtl + xbr) // 2, (ytl + ybr) // 2

            img = cv2.imread(output_file + annotation["@frame"] + ".jpg")
            cv2.line(img, (xbr, ytl), (xtl, ytl), (0, 0, 255), thickness=2)
            cv2.line(img, (xbr, ytl), (xbr, ybr), (0, 0, 255), thickness=2)
            cv2.line(img, (xbr, ybr), (xtl, ybr), (0, 0, 255), thickness=2)
            cv2.line(img, (xtl, ybr), (xtl, ytl), (0, 0, 255), thickness=2)
            cv2.putText(img, is_look, (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
            # cv2.imwrite("./save.jpg", img)

            # img = cv2.resize(img, (1920 // 2, 1080 // 2))
            # cv2.imshow("img", img)
            # cv2.waitKey(1)


# 输入字符类型的浮点数，返回int，用于cv2画图
def str_to_int(float_str):
    return int(float(float_str))


def main():
    for i in range(1, 300):
        video_id = "video_" + str(i).zfill(4)
        xml_path = "E:/CodeResp/pycode/DataSet/JAAD-JAAD_2.0/annotations/" + video_id + ".xml"
        output_data_path = "E:/CodeResp/pycode/DataSet/JAAD_image/" + video_id + "/"
        jaad_read(xml_path, output_data_path, video_id)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
