import pandas as pd
from sklearn.metrics import roc_auc_score

from log_config.log import logger


# 真实正值（TP）-表示该类为“真值”的次数，您的模型也表示它为“真值”。
# 真负数  （TN）-表示该类为假值的次数，您的模型也表示它为假值。
# 误报   （FP）-表示该类为假值，但您的模型表示为真值。
# 假阴性  （FN）-表示该类为“真值”的次数，但您的模型表示为“假值”。
def calculate_TP(y_true, y_predict):
    tp = 0
    for i, j in zip(y_true, y_predict):
        if i == j == 1:
            tp += 1
    return tp


def calculate_TN(y_true, y_predict):
    tn = 0
    for i, j in zip(y_true, y_predict):
        if i == j == 0:
            tn += 1
    return tn


def calculate_FP(y_true, y_predict):
    fp = 0
    for i, j in zip(y_true, y_predict):
        if i == 0 and j == 1:
            fp += 1
    return fp


def calculate_FN(y_true, y_predict):
    fn = 0
    for i, j in zip(y_true, y_predict):
        if i == 1 and j == 0:
            fn += 1
    return fn


# 精度 Precision
# 精度度量有助于我们理解识别阳性样本的正确性%。例如，我们的模型假设有80次是正的，我们精确地计算这80次中有多少次模型是正确的。
def calculate_precision(y_true, y_predict):
    tp = calculate_TP(y_true, y_predict)
    fp = calculate_FP(y_true, y_predict)
    if (tp + fp) == 0:
        return 0
    return tp / (tp + fp)


# 召回率 Recall
# 召回指标可帮助我们了解模型能够正确识别的所有地面真实正样本中正样本的百分比。 例如-假设数据中有100个阳性样本，我们计算出该100个样本中有多少个模型能够正确捕获。
def calculate_recall(y_true, y_predict):
    tp = calculate_TP(y_true, y_predict)
    fn = calculate_FN(y_true, y_predict)
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


# F1分数
# F1结合了Precision和Recall得分，得到一个单一的数字，可以帮助直接比较不同的模型。
# 可以将其视为P和R的谐波均值。谐波均值是因为与其他方式不同，它对非常大的值不敏感。 当处理目标倾斜的数据集时，我们通常考虑使用F1而不是准确性。
def calculate_F1(y_true, y_predict):
    p = calculate_precision(y_true, y_predict)
    r = calculate_recall(y_true, y_predict)
    if (p + r) == 0:
        return 0
    return 2 * p * r / (p + r)


# AUC-ROC
# AUC-ROC是用于二分类问题的非常常见的评估指标之一。 这是一条曲线，绘制在y轴的TPR（正确率）和x轴的FPR（错误率）之间，其中TPR和FPR定义为-
# TPR = TP/(TP +FN)
# FPR = FP/(TN + FP)
# 如果您注意到，TPR和Recall具有相同的表示形式，就像您正确分类了多少正确样本一样。 另一方面，FPR是被错误分类的负面示例的比例。
# ROC图总结了每个阈值的分类器性能。
# 因此，对于每个阈值，我们都有TPR和FPR的新混淆矩阵值，这些值最终成为ROC 2-D空间中的点。
# ROC曲线下的AUC（曲线下的面积）值越接近1，模型越好。
# 这意味着一般而言，对于具有较高AUC的每个阈值，我们的模型都比其他模型具有更好的性能。
# 仅用于二分类
def roc_auc(y_true, y_predict):
    return roc_auc_score(y_true, y_predict)


# Precision @ k
# Precision @ k是用于多标签分类设置的流行指标之一。 在此之下，我们计算给定示例的前k个预测，然后计算出这k个预测中有多少个实际上是真实标签。 我们将Precision @ k计算为-
# Precision@k = (# of correct predictions from k) / (# of items in k)

# log损失
# 当您遇到二分类问题时，log损失是相当不错的。 当您有一个模型输出概率时，该模型将使用该模型，该模型会根据预测与实际标签的偏差来考虑预测的不确定性。
# def calculate_log_loss(y, y_predict_probs):
#     log_loss = -1.0*(t*log(p) + (1-t)*(t*log(1-p))
#     return log_loss
# 在不平衡数据集的情况下，您还可以添加类权重来惩罚少数类相对于多数类的错误。在代码中，w1和w2分别对应正类和负类的权重。
# def calculate_log_loss_weighted(y_true, y_predict):
#     log_loss = -1.0*(w1*t*log(p) + w2*(1-t)*(t*log(1-p))
#     return log_loss

# Brier分数
# 当任务本质上是二元分类时，通常使用Brier分数。 它只是实际值和预测值之间的平方差。 对于N组样本，我们将其取平均值。
def brier_score(y_true, y_predict):
    s = 0
    for i, j in zip(y_true, y_predict):
        s += (j - i) ** 2
    return s * (1 / len(y_true))


# 准确率 Accuracy
# 准确使人们对模型的运行方式有了整体认识。 但是，如果使用不正确，它很容易高估这些数字。
# 例如-如果类标签的分布偏斜，则仅预测多数类会给您带来高分（高估性能），而对于平衡类而言，准确性更有意义。
def calculate_accuracy(y_true, y_predict):
    tp = calculate_TP(y_true, y_predict)
    tn = calculate_TN(y_true, y_predict)
    fp = calculate_FP(y_true, y_predict)
    fn = calculate_FN(y_true, y_predict)
    if (tp + tn + fp + fn) == 0:
        return 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print("当前方法的检测准确率为%0.3f%%" % (accuracy * 100))
    return accuracy


def calculate_all(y_true, y_predict):
    calculate_name = [calculate_TP, calculate_TN, calculate_FP, calculate_FN,
                      calculate_precision, calculate_recall, calculate_F1, roc_auc, brier_score, calculate_accuracy]
    for name in calculate_name:
        logger.info("数值:%f, 指标:%s" % (name(y_true, y_predict), name.__name__))


if __name__ == "__main__":
    y = [1, 1, 1, 1, 0]
    y_pre = [0, 1, 1, 1, 0]
    video_label = pd.read_csv("../train/trained_model/video/Forest_image_y_pred.csv", header=None, sep=',',
                              encoding='utf-8').values
    image_label = pd.read_csv("../train/trained_model/image/Forest_image_y_pred.csv", header=None, sep=',',
                              encoding='utf-8').values
    calculate_all(video_label, image_label)
