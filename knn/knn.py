from numpy import *
import operator
import  matplotlib
import  matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

group,labels = createDataSet()

# inX: 向量, dataSet: 数据集, labels: 标签, k: 取近邻个数
def classify0(inX, dataSet, labels, k):
   m = dataSet.shape[0]
   diffMat = tile(inX, (m, 1)) - dataSet
   sq = diffMat ** 2
   sum = sq.sum(axis=1)
   sortIns = sum.argsort()
   classCount = {}
   for i in range(k):
       label = labels[sortIns[i]]
       classCount[label] = classCount.get(label, 0) + 1
   # print(classCount)
   sortClassCount = sorted(classCount.items(),
                           key=operator.itemgetter(0),
                           reverse=True)
   return sortClassCount[0][0]

# print(classify0([0,0], group, labels, 3))

def file2matrix(filename):
    fr = open(filename)
    arrayOfLine = fr.readlines()
    length = len(arrayOfLine)
    returnMat = zeros((length, 3))
    labels = []
    index = 0
    for line in arrayOfLine:
        line = line.strip()
        lineArray = line.split('\t')
        returnMat[index,:] = lineArray[0:3]
        labels.append(int(lineArray[-1]))
        index += 1
    return returnMat, labels



def autoNorm(dataSet):
    minVals = dataSet.min(0) #取第一列特征最小值
    maxVals = dataSet.max(0) #取第一列特征最大值
    ranges = maxVals - minVals #范围
    normDataSet = zeros(shape(dataSet)) #0占位矩阵
    m = dataSet.shape[0] #训练集长度
    normDataSet = dataSet - tile(minVals, (m, 1)) #计算
    normDataSet = normDataSet / tile(ranges, (m, 1)) #计算
    return  normDataSet, ranges, minVals



def datingClassTest ():
    hoRatio = 0.10
    datingDataSet, datingLabels = file2matrix('datingDataSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataSet)
    m = normMat.shape[0]
    numTestVecs = m * hoRatio
    errorCount = 0.0
    numTestVecs = int(numTestVecs)
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("分类器返回", classifierResult, ", 正确答案是", datingLabels[i])
        if(classifierResult != datingLabels[i]): errorCount += 1.0
    print('总错误率是', (errorCount / float(numTestVecs)))



datingClassTest()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(returnMat[:, 0], returnMat[:, 1])
# plt.show()