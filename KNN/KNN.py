import numpy as np
import operator
from os import listdir

# 实现对文件数据的读取和预处理
def fileToMatrix(filename):
    # 类别含义与编码的映射关系
    loveClasses = {"largeDoses": 3, "smallDoses": 2, "didntLike": 1}
    # 读取文件获得行数n(即样本个数n)
    file = open(filename)
    fileData = file.readlines()
    numOfLines = len(fileData)
    # 建立数据矩阵(n*3)和对应的标记向量(n*1)
    dataMatrix = np.zeros((numOfLines, 3))
    classLabelVector = []
    index = 0
    for line in fileData:
        # 处理一行的字符串
        line = line.strip()
        lineList = line.split('\t')
        # 前三个数据为样本数据,最后一个数据为标记
        dataMatrix[index, 0:3] = lineList[0:3]
        # 判断最后一个数据的组织格式
        if lineList[-1].isdigit():
            classLabelVector.append(lineList[-1])
        else:
            classLabelVector.append(loveClasses.get(lineList[-1]))
    return dataMatrix, classLabelVector
# KNN算法的实现
def knn(inX, dataSet, labels, k):
    # 获得输入数据集的矩阵行数,即数据集中数据个数
    dataSetSize = dataSet.shape[0]
    # 根据数据集个数构造inX的矩阵并减去数据集
    diffMat = np.tile(inX, (dataSetSize - 1)) - dataSet
    # 计算欧式距离
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    # 选取距离最近的前k个样本获取其排序前的索引
    sortedDistance = distance.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistance[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter, reverse=True)
    return sortedClassCount[0][0]







