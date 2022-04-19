import cv2
import xmltodict


def XMLRead(xml_file_path, output_file, video_id):
    with open(xml_file_path) as xml_file:
        # xmltodict.parse()方法可以将xml数据转为python中的dict字典数据
        parser_data = xmltodict.parse(xml_file.read())
        annotations = parser_data["annotations"]
        track_box = annotations["track"]["box"]
        for annotation in track_box:
            xbr = str_to_int(annotation["@xbr"])  # 左上角，右下角 (top-left, bottom-right)
            is_look = annotation["attribute"][2]["#text"]
            xtl = str_to_int(annotation["@xtl"])
            ybr = str_to_int(annotation["@ybr"])
            ytl = str_to_int(annotation["@ytl"])
            x_mid, y_mid = (xtl + xbr) // 2, (ytl + ybr) // 2

            img = cv2.imread(output_file + annotation["@frame"] + ".jpg")
            cv2.line(img, (xbr, ytl), (xtl, ytl), (0, 0, 255), thickness=2)
            cv2.line(img, (xbr, ytl), (xbr, ybr), (0, 0, 255), thickness=2)
            cv2.line(img, (xbr, ybr), (xtl, ybr), (0, 0, 255), thickness=2)
            cv2.line(img, (xtl, ybr), (xtl, ytl), (0, 0, 255), thickness=2)
            cv2.putText(img, is_look, (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
            cv2.imwrite("./save.jpg", img)
            break
        xml_file.close()


# 输入字符类型的浮点数，返回int，用于cv2画图
def str_to_int(float_str):
    return int(float(float_str))


def main():
    video_id = "video_0014"
    xml_path = "E:/CodeResp/pycode/DataSet/JAAD-JAAD_2.0/annotations/" + video_id + ".xml"
    output_data_path = "E:/CodeResp/pycode/DataSet/JAAD_image/" + video_id + "/"
    XMLRead(xml_path, output_data_path, video_id)


if __name__ == "__main__":
    main()
