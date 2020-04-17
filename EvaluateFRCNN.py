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
    trueLabels = pandas.read_csv('D:/Pilot_Study/' + videoName + '_adapted.csv')
    trueLabels.head()
    predictedLabels = pandas.read_csv('D:/Pilot_Study/results_images/'+videoName+'_testDataBoundingBoxes.csv')
    predictedLabels.head()
    return(trueLabels,predictedLabels)

def getAccuracy(trueLabels,predictedLabels):
    totalFrameCount = len(trueLabels)
    truePositives = 0
    trueNegatives = 0
    falsePositives = 0
    falseNegatives = 0
    for i in range(len(trueLabels)):
        if trueLabels['class_name'][i] == 'ultrasound' and predictedLabels['label'][i] == 'ultrasound':
            truePositives +=1
        elif trueLabels['class_name'][i] == 'none' and predictedLabels['label'][i] == 'none':
            trueNegatives +=1
        elif trueLabels['class_name'][i] == 'ultrasound' and predictedLabels['label'][i] == 'none':
            falseNegatives += 1
        elif trueLabels['class_name'][i] == 'none' and predictedLabels['label'][i] == 'ultrasound':
            falsePositives += 1
    accuracy = (truePositives + trueNegatives) / totalFrameCount
    print('Accuracy: {}\nTrue positives: {}\nTrue negatives: {}\nFalse positives: {}\nFalse negatives: {}\nTotal frame count: {}'.format(accuracy,truePositives,trueNegatives,falsePositives,falseNegatives,totalFrameCount))
    return (accuracy,truePositives,trueNegatives,falsePositives,falseNegatives,totalFrameCount)

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
    videos = ['MS01-20200210-133541']
    #videos = ['MS01-20200210-132740','MS01-20200210-133541','MS01-20200210-134522','MS01-20200210-135109','MS01-20200210-135709']
    #videoLengths = [583.88,315.05,284.73,234.53,219.39]
    videoLengths = [315.05]
    dataFolder = 'D:/Pilot_Study/Segmented_videos/'
    imageFolder = '/training_photos/task_labels/wholevideo'
    resultsData = pandas.DataFrame(columns = ['video_name','accuracy','true_positives','true_negatives','false_positives','false_negatives','total_frame_count','mean_euclidean_distance','std_deviation','scan_time','path_length','true_path_length'])
    for i in range(len(videos)):
        print(videos[i])
        trueLabels,predictedLabels = readCSV(videos[i])
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
    resultsData.to_csv('D:/Pilot_Study/results_images/performanceResults.csv')

main()



