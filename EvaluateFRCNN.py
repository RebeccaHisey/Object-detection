import os
import csv
import numpy
import math
import cv2
import sklearn.metrics
from matplotlib import pyplot as plt
import pandas
import SimpleITK

def readCSV(videoName):
    trueLabels = pandas.read_csv('D:/Pilot_Study/Olivia_Annotations/crossvalidation/unbalanced_for_testing/' + videoName + '_adapted.csv')
    trueLabels.head()
    predictedLabels = pandas.read_csv('D:/Pilot_Study/results_images/Olivia/'+videoName+'_testDataBoundingBoxes.csv')
    predictedLabels.head()
    return(trueLabels,predictedLabels)

def convertNametoIndex(name):
    if name == 'anesthetic':
        index = 0
    elif name == 'catheter':
        index= 1
    elif name == 'dilator':
        index  = 2
    elif name == 'guidewire':
        index  = 3
    elif name == 'guidewire_casing':
        index  = 4
    elif name == 'none':
        index  = 5
    elif name == 'scalpel':
        index  = 6
    elif name == 'syringe':
        index  = 7
    elif name == 'ultrasound':
        index  = 8
    return index

def getAccuracy(trueLabels,predictedLabels):
    tLabels =[]
    pLabels =[]
    checkedFiles =[]
    for i in range(len(predictedLabels)):
        currentFile = predictedLabels['filePath'][i]
        if currentFile not in checkedFiles:
            checkedFiles.append(predictedLabels['filePath'][i])
            allTrueBBoxes = trueLabels.loc[trueLabels['filePath'] == currentFile]
            allPredBBoxes = predictedLabels.loc[predictedLabels['filePath'] == currentFile]
            trueRowIndexes = allTrueBBoxes.index
            predRowIndexes = allPredBBoxes.index
            if len(predRowIndexes)> len(trueRowIndexes):
                for j in predRowIndexes:
                    Plabel = allPredBBoxes['label'][j]
                    Tlabel = 'none'
                    for k in trueRowIndexes:
                        if allTrueBBoxes['class_name'][k] == Plabel:
                            Tlabel = allTrueBBoxes['class_name'][k]
                    Tlabel = convertNametoIndex(Tlabel)
                    Plabel = convertNametoIndex(Plabel)
                    tLabels.append(Tlabel)
                    pLabels.append(Plabel)
            else:
                for j in trueRowIndexes:
                    Tlabel = allTrueBBoxes['class_name'][j]
                    Plabel = 'none'
                    for k in predRowIndexes:
                        if allPredBBoxes['label'][k] == Tlabel:
                            Plabel = allPredBBoxes['label'][k]
                    Tlabel = convertNametoIndex(Tlabel)
                    Plabel = convertNametoIndex(Plabel)
                    tLabels.append(Tlabel)
                    pLabels.append(Plabel)
    tLabels = numpy.array(tLabels)
    pLabels = numpy.array(pLabels)
    confmat = sklearn.metrics.confusion_matrix(tLabels,pLabels)
    print(confmat)
    diagonalEntries = []
    for i in range(0,confmat.shape[0]):
        diagonalEntries.append(confmat[i][i])
    diagonalEntries = numpy.array(diagonalEntries)
    accuracy = numpy.sum(diagonalEntries) / numpy.sum(confmat)
    totalFrameCount = len(pLabels)
    return (accuracy,totalFrameCount)

def getCenterPoints(trueLabels,predictedLabels):
    centerPoints = pandas.DataFrame(columns=['filePath','x_center_true','y_center_true','x_center_pred','y_center_pred','distance'])
    for i in range(len(trueLabels)):
        if trueLabels['class_name'][i] != 'none' and predictedLabels['label'][i]!='none':
            xCenterTrue = trueLabels['x_min'][i] + round((trueLabels['x_max'][i] - trueLabels['x_min'][i])/2)
            yCenterTrue = trueLabels['y_min'][i] + round((trueLabels['y_max'][i] - trueLabels['y_min'][i]) / 2)
            xCenterPred = predictedLabels['xmin'][i] + round((predictedLabels['xmax'][i] - predictedLabels['xmin'][i]) / 2)
            yCenterPred = predictedLabels['ymin'][i] + round((predictedLabels['ymax'][i] - predictedLabels['ymin'][i]) / 2)
            distance = math.sqrt((xCenterTrue-xCenterPred)**2 + (yCenterTrue-yCenterPred)**2)
            centerPoints = centerPoints.append({'filePath':predictedLabels['filePath'][i],'x_center_true':xCenterTrue,'y_center_true':yCenterTrue,'x_center_pred':xCenterPred,'y_center_pred':yCenterPred,'distance':distance},ignore_index=True)
    centerPoints.to_csv('D:/Pilot_Study/results_images/'+trueLabels['video_name'][0]+'_centerPointDistances.csv')
    return centerPoints

def getMeanDistance(trueLabels,predictedLabels):
    centerPoints = getCenterPoints(trueLabels,predictedLabels)
    meanDistance = centerPoints['distance'].mean()
    stdDev = centerPoints['distance'].std()
    print ('Mean average distance: {} +/- {}'.format(meanDistance,stdDev))
    return (meanDistance,stdDev)

def getScanTimeEstimate(truePositives,falsePositives,frameCount,videoLength):
    numFramesOnScreen = truePositives + falsePositives
    secPerFrame = videoLength / frameCount
    scanTime = numFramesOnScreen * secPerFrame
    print('Estimated scan time: {}s'.format(scanTime))
    return scanTime

def getPathLengthEstimate(predictedLabels,trueLabels,videoPath):
    fileNames = os.listdir(videoPath)
    dateCreated = [(os.path.getmtime(os.path.join(videoPath, fileName)), fileName) for fileName in fileNames]
    sequentialFiles = [fileName for _, fileName in sorted(dateCreated)]
    previousXCenter = None
    previousYCenter = None
    totalPathLength = 0
    previousTrueX = None
    previousTrueY = None
    totalTruePath = 0
    for file in sequentialFiles:
        i=0
        imageFileFound = False
        while i <(len(predictedLabels)) and not imageFileFound:
            imgFileName = os.path.basename(predictedLabels['filePath'][i])
            if imgFileName == file:
                imageFileFound = True
                if predictedLabels['label'][i] != 'none':
                    if previousXCenter == None:
                        previousXCenter = predictedLabels['xmin'][i] + ((predictedLabels['xmax'][i] - predictedLabels['xmin'][i]) / 2)
                        previousYCenter = predictedLabels['ymin'][i] + ((predictedLabels['ymax'][i] - predictedLabels['ymin'][i]) / 2)
                    else:
                        currXCenter = predictedLabels['xmin'][i] + ((predictedLabels['xmax'][i] - predictedLabels['xmin'][i]) / 2)
                        currYCenter = predictedLabels['ymin'][i] + ((predictedLabels['ymax'][i] - predictedLabels['ymin'][i]) / 2)
                        distance = math.sqrt((currXCenter-previousXCenter)**2 + (currYCenter-previousYCenter)**2)
                        totalPathLength += distance
                        previousXCenter = currXCenter
                        previousYCenter = currYCenter
                if trueLabels['class_name'][i] != 'none':
                    if previousTrueX == None:
                        previousTrueX = trueLabels['x_min'][i] + ((trueLabels['x_max'][i] - trueLabels['x_min'][i]) / 2)
                        previousTrueY = trueLabels['y_min'][i] + ((trueLabels['y_max'][i] - trueLabels['y_min'][i]) / 2)
                    else:
                        currXTrue = trueLabels['x_min'][i] + ((trueLabels['x_max'][i] - trueLabels['x_min'][i]) / 2)
                        currYTrue = trueLabels['y_min'][i] + ((trueLabels['y_max'][i] - trueLabels['y_min'][i]) / 2)
                        distance = math.sqrt((currXTrue-previousTrueX)**2 + (currYTrue-previousTrueY)**2)
                        totalTruePath += distance
                        previousTrueX = currXTrue
                        previousTrueY = currYTrue
            else:
                i +=1

    print('Estimated path length: {}'.format(totalPathLength))
    print('True path length: {}'.format(totalTruePath))
    return (totalPathLength,totalTruePath)

def main():
    videos = ['MS03-20200213-152826','MS03-20200213-153647','MS03-20200213-154347','MS03-20200213-155250','MS03-20200213-155823']
    #videos = ['MS01-20200210-132740','MS01-20200210-133541','MS01-20200210-134522','MS01-20200210-135109','MS01-20200210-135709']
    #videoLengths = [583.88,315.05,284.73,234.53,219.39]
    videoLengths = [315.05]
    dataFolder = 'D:/Pilot_Study/Segmented_videos/'
    imageFolder = '/training_photos/task_labels/wholevideo'
    resultsData = pandas.DataFrame(columns = ['video_name','accuracy','total_frame_count','mean_euclidean_distance','std_deviation','scan_time','path_length','true_path_length'])
    for i in range(len(videos)):
        print(videos[i])
        trueLabels,predictedLabels = readCSV(videos[i])
        (accuracy,totalFrameCount) = getAccuracy(trueLabels,predictedLabels)
        print (accuracy)
        '''
        (accuracy, truePositives, trueNegatives, falsePositives, falseNegatives, totalFrameCount) = getAccuracy(trueLabels,predictedLabels)
        (meanDistance,stdDev) = getMeanDistance(trueLabels,predictedLabels)
        scanTime = getScanTimeEstimate(truePositives,falsePositives,totalFrameCount,videoLengths[i])
        imgPath = dataFolder + videos[i] + imageFolder
        (pathLength,truePathLength) = getPathLengthEstimate(predictedLabels,trueLabels,imgPath)
        resultsData = resultsData.append({'video_name':videos[i],
                                         'accuracy':accuracy,
                                         'true_positives':truePositives,
                                         'true_negatives':trueNegatives,
                                         'false_positives':falsePositives,
                                         'false_negatives':falseNegatives,
                                         'total_frame_count':totalFrameCount,
                                         'mean_euclidean_distance':meanDistance,
                                         'std_deviation':stdDev,
                                         'scan_time':scanTime,
                                         'path_length':pathLength,
                                         'true_path_length':truePathLength},ignore_index=True)
        '''
    #resultsData.to_csv('D:/Pilot_Study/results_images/performanceResults_balanced.csv')

main()



