import numpy as np
import operator
from os import listdir

# 实现对文件数据的读取和预处理
def fileToMatrix(filename):
    '''
    :param filename: 数据文件
    :return: 结构化的数据矩阵
    '''
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

# 对数据进行归一化处理,使得所有数据都在同一个数量级上
def autoNorm(dataSet):
    '''
    :param dataSet: 数据矩阵
    :return: 归一化后的数据矩阵
    归一化算法 : x -> Y : Y = (x - xMin) / (xMax - xMin)
    '''
    # 求出每一列的最小最大值
    minVals = dataSet.min(0);
    maxVals = dataSet.max(0);
    ranges = maxVals - minVals;
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    # 减去最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 除以范围
    normDataSet = dataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# KNN算法的实现
def knn(inX, dataSet, labels, k):
    # 获得输入数据集的矩阵行数,即数据集中数据个数
    dataSetSize = dataSet.shape[0]
    # 根据数据集个数构造inX的矩阵并减去数据集
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
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
    maxClassCount = max(classCount, key=classCount.get)
    return maxClassCount

def datingClassTest():
    # 选取10%作为测试集
    testRatio = 0.5
    datingDataMat, datingLabels = fileToMatrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTest = int(m * testRatio)
    errorCount = 0
    for i in range(numTest):
        classifierResult = knn(normMat[i, :], normMat[numTest:m, :], datingLabels[numTest:m], 100)
        print("the predict is %d, the truth is %d" % (int(classifierResult), int(datingLabels[i])))
        if classifierResult != datingLabels[i] : errorCount += 1
    print("total error rate is %f" % (errorCount / numTest))


if __name__ == '__main__':
    datingClassTest()






