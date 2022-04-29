# 数据上采样、下采样，解决数据不平衡问题
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NearMiss


# 朴素随机过采样（上采样） ,从少数类的样本中进行随机采样来增加新的样本
def naive_random_over_sample(data, label):
    ros = RandomOverSampler(random_state=0, shrinkage=0.1)
    x_resampled, y_resampled = ros.fit_resample(data, label)
    return x_resampled, y_resampled


# 过采样的改进方式：
# (1)Synthetic Minority Oversampling Technique(SMOTE)
# (2)Adaptive Synthetic (ADASYN)
# SMOTE算法的基本思想是对少数类样本进行分析并根据少数类样本人工合成新样本添加到数据集中
# 1、对于少数类中每一个样本x，计算该点与少数类中其他样本点的距离，得到最近的k个近邻（即对少数类点进行KNN算法）。
# 2、根据样本不平衡比例设置一个采样比例以确定采样倍率，对于每一个少数类样本x，从其k近邻中随机选择若干个样本，假设选择的近邻为x’。
# 3、对于每一个随机选出的近邻x’，分别与原样本按照如下的公式构建新的样本：
# 但是SMOTE算法缺点也十分明显：一方面是增加了类之间重叠的可能性（由于对每个少数类样本都生成新样本，因此容易发生生成样本重叠(Overlapping)的问题），
# 另一方面是生成一些没有提供有益信息的样本。
def smote_sample(data, label):
    ros = SMOTE(sampling_strategy="auto", k_neighbors=7, n_jobs=-1)
    x_resampled, y_resampled = ros.fit_resample(data, label)
    return x_resampled, y_resampled


def svm_smote_sample(data, label):
    ros = SVMSMOTE(sampling_strategy="auto", random_state=0, k_neighbors=5, n_jobs=-1, m_neighbors=10,
                   svm_estimator=None, out_step=0.5)
    x_resampled, y_resampled = ros.fit_resample(data, label)
    return x_resampled, y_resampled


# Borderline-SMOTE,目前存在bug，无法得到计算结果,也可能是计算过慢
def cluster_centroids(data, label):
    ros = ClusterCentroids(random_state=42)
    x_resampled, y_resampled = ros.fit_resample(data, label)
    return x_resampled, y_resampled


# 自适应综合采样（ADASYN）首先根据最终的平衡程度设定总共需要生成的新少数类样本数量 ，然后为每个少数类样本x计算分布比例。
def adasyn(data, label):
    x_resampled, y_resampled = ADASYN().fit_resample(data, label)
    return x_resampled, y_resampled


# 下采样
# 朴素随机欠采样（下采样）,与过采样相反，欠采样是从多数类样本中随机选择少量样本，再合并原有少数类样本作为新的训练数据集。
def naive_random_under_sample(data, label):
    ros = RandomUnderSampler(sampling_strategy=0.5,random_state=0)
    x_resampled, y_resampled = ros.fit_resample(data, label)
    return x_resampled, y_resampled


#  (i) estimator是选择使用的分类器；(ii) n_max_subset控制的是子集的个数；(iii)  bootstrap决定是有放回还是无放回的随机采样。
def balance_cascade(data, label):
    bc = BorderlineSMOTE(
        sampling_strategy="auto", random_state=None, k_neighbors=5, n_jobs=-1, m_neighbors=10, kind="borderline-1")
    x_resampled, y_resampled = bc.fit_resample(data, label)
    return x_resampled, y_resampled


def near_miss(data, label):
    ee = NearMiss(random_state=0, version=1)
    x_resampled, y_resampled = ee.fit_resample(data, label)
    return x_resampled, y_resampled


def default(all_data, all_labels):  # 默认情况下执行的函数
    print('未选择数据采样函数')


def data_resample(method_id: int, x_train, y_train):
    name_list = ["naive_random_over_sample", "smote_sample", "cluster_centroids", "adasyn", "naive_random_under_sample",
                 "balance_cascade", "near_miss", ]
    sample_method = {"naive_random_over_sample": naive_random_over_sample,
                     "smote_sample": smote_sample,
                     "cluster_centroids": cluster_centroids,
                     "adasyn": adasyn,
                     "naive_random_under_sample": naive_random_under_sample,
                     "balance_cascade": balance_cascade,
                     "near_miss": near_miss,
                     }
    method = name_list[method_id]  # 获取选择
    x_resampled, y_resampled = sample_method.get(method, default)(x_train, y_train)  # 执行对应的函数，如果没有就执行默认的函数
    return x_resampled, y_resampled
