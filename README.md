# IRBOPP
Intention recognition based on pedestrian perception\
基于行人感知的意图识别\
# TODO
1、实现行人感知\
2、实现姿势识别\
3、实现头部方向分类
# JAAD 注释是根据视频剪辑名称组织的。共有三种标签，行人（带有行为注释的样本）、peds（距离较远且不与驾驶员互动的旁观者）和人（行人群体）。
# 每个行人都有一个唯一的 id，形式为0_<video_id>_< pedestrian_number>。
# 带有行为注释的行人在他们的 id 末尾有一个字母“b”，例如0_1_3b。人们的注释也遵循相同的模式，除了以字母“p”结尾，例如0_5_2p。
# 所有样本都使用两点坐标（左上角，右下角）用边界框进行注释[x1, y1, x2, y2]。边界框具有相应的遮挡标签。遮挡值是 0（无遮挡）、1（部分遮挡 >25%）或 2（完全遮挡 >75%）。
# 根据它们的类型，注释分为 5 组：
# 注释：这些包括视频属性（一天中的时间、天气、位置）、行人边界框坐标、遮挡信息和活动（例如步行、看）。这些活动仅提供给一部分行人。这些注释是每个标签每帧一个。
# 属性（仅带有行为注释的行人）：这些包括有关行人的人口统计、交叉点、交叉点特征等的信息。这些注释是每个行人一个。
# 外观（仅限高能见度的视频）：这些包括有关行人外观的信息，例如姿势、服装、物体 carreid（请参阅_get_ped_appearance()更多细节）。这些注释是每个行人每帧一个。
# 交通：这些为每一帧提供有关交通的信息，例如标志、交通灯。这些注释是每帧一个。
# 车辆：这些是车辆动作，例如，每帧快速移动、加速。