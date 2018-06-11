import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])

    return returnVect

def handwritingClassCount():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

    neigh = kNN(n_neighbors=3,algorithm='auto')
    neigh.fit(trainingMat,hwLabels)

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        vectUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        classifierResult = neigh.predict(vectUnderTest)
        print("分类结果%d真实结果%d" % (classifierResult,classNumber))

        if classifierResult != classNumber:
            errorCount += 1.0

    print("总共错了%d错误率为%s%%" % (errorCount,errorCount/mTest*100))

if __name__ == '__main__':
    handwritingClassCount()