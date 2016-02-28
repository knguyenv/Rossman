import csv
import sklearn
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans


#Strip out the sales figures into an array, and delete the number of customer field
def PreProcessing(inputArr):
    inputDict = {}
    salesDict = {}
    salesCluster = {}
    StoreOpenDict = {}

    for row in inputArr:
        isOpened = int(row[5])
        dateStr = row[2]
        if not(StoreOpenDict.has_key(dateStr)):
            StoreOpenDict[dateStr] = 0
        if isOpened == 1:
            StoreOpenDict[dateStr] += 1  #record number of store opened on that date


    for row in inputArr:
        salesNumber = int(row.pop(3))     #get and remove sales number
        row.pop(3)     #remove number of customers
        if row[5] in ('a', 'b', 'c'): row[5] = ord(row[5])
        else: row[5] = int(row[5])
        row[6] = int(row[6])
        storeId = int(row.pop(0))      #remove store ID

        dateStr = row[1]
        dateFeatures = row[1].split('-')
        row[1] = int(dateFeatures[1])   #Get month from the date
        for col in range(0,5): row[col] = int(row[col])

        if row[2] > 0:
            row[2] = int(dateFeatures[0])   #Get year from the date
            if not(inputDict.has_key(storeId)):
                inputDict[storeId] = []
                salesDict[storeId] = []
                salesCluster[storeId] = []
            inputDict[storeId].append(row)
            salesDict[storeId].append(salesNumber)

            #only if all stores opened on this date
            if StoreOpenDict.has_key(dateStr) and StoreOpenDict[dateStr] == 1115:
                salesCluster[storeId].append(salesNumber)

        #inputDict - 0:DayOfWeek,  1:month,  2:year,  3:Promo,  4:StateHoliday,  5:SchoolHoliday

    return inputDict, salesDict, salesCluster



def PreprocessTestSet(testArr):
    testDict = {}
    for row in testArr:
        for col in range(0, 8):
            if len(row[col]) == 0: row[col] = 0
        storeId = int(row.pop(1))   #get storeID
        dateFeatures = row[2].split('-')
        row[2] = int(dateFeatures[1])   #get month from date
        row.insert(3, int(dateFeatures[0]))   #get year from date

        if row[6] in ('a', 'b', 'c'): row[6] = ord(row[6])
        else: row[6] = int(row[6])
        for col in range(0, 8): row[col] = int(row[col])
        if not(testDict.has_key(storeId)):
            testDict[storeId] = []
        testDict[storeId].append(row)
        #print row[0]
        #0:Id,  1:DayOfWeek,  2:month,  3:year,  4:Open,  5:Promo,  6:StateHoliday,  7:SchoolHoliday
    return testDict

#Calculate the Root Mean Square Percentage Error
def CalculateScore(y_real, y_predict):
    score = 0
    for i in range (0, len(y_real)):
        real = y_real[i]
        predict = y_predict[i]
        score += math.pow((real-predict)/real, 2)
    score = math.sqrt(score/len(y_real))
    return score


def PlotImportantFeatures(forestModel, size):
    # Build a classification task using 3 informative features
    X, y = make_classification(n_samples=size,
                               n_features=6,
                               n_informative=3,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=2,
                               random_state=0,
                               shuffle=False)

    importances = forestModel.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forestModel.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

#Build Random Forest model for each Cluster
def BuildForest(labelNr):
    score = 0
    inputArr = []
    salesArr = []
    testArr = []
    for Id in range(1, 1116):
        if testSet.has_key(Id) and ClusterLabels[Id-1] == labelNr:
            inputArr = inputArr + trainSet[Id]
            salesArr = salesArr + salesFigure[Id]
            testArr = testArr + testSet[Id]

    dataSet = np.array(inputArr)
    salesSet = np.array(salesArr)
    if len(dataSet) == 0:
        print 'Finished cluster ' + str(labelNr)
        return   #quit if there is no data to build Random forest

    clf = RandomForestRegressor(n_estimators=100)
    clf.fit(dataSet, salesSet)
    """
    kf = KFold(len(inputArr), n_folds=5)
    sklearn.cross_validation.KFold(n=len(inputArr), n_folds=5, shuffle=False, random_state=None)
    for train_index, test_index in kf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = dataSet[train_index], dataSet[test_index]
        y_train, y_test = salesSet[train_index], salesSet[test_index]
        score += CalculateScore(clf.predict(X_test), y_test)
    score = score / 5
    """
    #PlotImportantFeatures(clf, len(inputArr))

    predictionResult = []
    predictionInput = copy.deepcopy(testArr)
    for row in predictionInput:
        row.pop(0)     #remove row Id
        row.pop(3)     #remove store Open
    X = np.array(predictionInput)
    predictionResult = clf.predict(X)

    for index in range(0, len(testArr)):
        test = testArr[index]
        rowId = test.pop(0)    #remove row Id
        if test[3]==0: testPrediction[rowId] = 0   #Store closed --> zero sales
        else:
            testPrediction[rowId] = predictionResult[index]

    #print testPrediction
    print 'Finished cluster ' + str(labelNr)

#Make Clusters and perform labeling for each store
def MakeCluster():
    Labels = []
    k_means = KMeans(init='k-means++', n_clusters=NrOfCluster, n_init=10)
    X = []
    for key in salesCluster.keys():
        X.append(salesCluster[key])
    k_means.fit(X)
    print k_means.inertia_
    for index in range(0, len(k_means.labels_)):
        Labels.append(k_means.labels_[index])
    return Labels

#Print sales prediction results to file
def PrintSalesPrediction():
    file = open('prediction.csv', 'w+')
    file.write('id,sales' + '\n')
    keyList = sorted(testPrediction.keys())

    for key in keyList:
        salesNum = testPrediction[key]
        file.write(str(key) + ',' + str(salesNum) + '\n')
    file.close()

csvReader = csv.reader(open('train.csv', 'rb'), delimiter=',', quotechar='"')
input = []
for row in csvReader:
    input.append(row)

csvReader = csv.reader(open('test.csv', 'rb'), delimiter=',', quotechar='"')
test = []
for row in csvReader:
    test.append(row)

labels = input.pop(0)
labels.pop(3), labels.pop(3), labels.pop(0)
#labels[1] = "month", labels[2] = "year"
trainSet, salesFigure, salesCluster = PreProcessing(input)

testLabels = test.pop(0)
testSet = PreprocessTestSet(test)
testPrediction = {}

startLabel = 0
endLabel = 1116
NrOfCluster = 1115
ClusterLabels = MakeCluster()

for labelNr in range(startLabel, endLabel):
    BuildForest(labelNr)
PrintSalesPrediction()


