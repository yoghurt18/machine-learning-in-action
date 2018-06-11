import numpy as np
import operator
from os import listdir

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistances = distances.argsort()

    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)

    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i + j] = int(lineStr[j])

    return returnVect

def handwritingClassTest():
    hwlabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumber = int(fileStr.split('_')[0])
        hwlabels.append(classNumber)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumber = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,
                                     trainingMat,hwlabels,3)
        print("分类结果%d真实结果%d" % (classifierResult,classNumber))

        if classifierResult != classNumber:
            errorCount += 1.0
    print("总共错了%d数据错误率为%f%%" % (errorCount,errorCount/mTest))


if __name__ == '__main__':
    handwritingClassTest()