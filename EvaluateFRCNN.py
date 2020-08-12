import os
import csv
import numpy
import math
import cv2
import sklearn.metrics
from matplotlib import pyplot as plt
import pandas
from scipy import optimize
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

def getIOU(bbox1,bbox2):
    [xmin1,ymin1,xmax1,ymax1] = bbox1
    [xmin2, ymin2, xmax2, ymax2] = bbox2
    x_left = max(xmin1,xmin2)
    x_right = min(xmax1,xmax2)
    y_top = max(ymin1,ymin2)
    y_bottom = min(ymax1,ymax2)

    intersection = (x_right - x_left)*(y_bottom-y_top)
    bbox1_area = (xmax1-xmin1)*(ymax1-ymin1)
    bbox2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    union = float(bbox1_area + bbox2_area - intersection)

    intersectionOverUnion = intersection / union
    return intersectionOverUnion

def smoothPrecision(precision):
    for i in range(0,len(precision)):
        precision[i] = max(precision[i:])
    return precision

def getMAP(trueLabels,predLabels):
    toolAcronyms = ['a','c','d','g','gc','sc','sy']
    tools = ['anesthetic','catheter','dilator','guidewire','guidewire_casing','scalpel','syringe']
    classAP = []
    toolPrecision =[]
    toolRecall = []
    j=0
    for tool in tools:
        precision = []
        recall = []
        (rankedSamples,totalPosSamples) = getRankedToolPredictions(trueLabels,predLabels,tool,toolAcronyms[j])
        j+=1
        truePositives = 0
        falsePositives = 0
        falseNegatives = 0
        for i in range(len(rankedSamples)):
            if rankedSamples['true'][i] == 1 and rankedSamples['pred'][i]==1:
                truePositives += 1
            elif rankedSamples['true'][i] == 0 and rankedSamples['pred'][i] == 1:
                falsePositives += 1
            elif rankedSamples['true'][i] == 1 and rankedSamples['pred'][i] == 0:
                falseNegatives += 1
            if not truePositives+falsePositives == 0:
                precision.append(truePositives / float(truePositives + falsePositives))
                recall.append(truePositives / float(totalPosSamples))

        smoothedPrecision = numpy.array(smoothPrecision(precision))
        recall = numpy.array(recall)
        if precision !=[]:
            #
            auc = sklearn.metrics.auc(recall, numpy.array(precision))
        else:
            auc = 0.0
        toolPrecision.append(smoothedPrecision)
        toolRecall.append(recall)
        classAP.append(auc)
        print('{}: {}'.format(tool,auc))
    plt.plot(toolRecall[0],toolPrecision[0], 'm', label='anesthetic')
    plt.plot(toolRecall[1], toolPrecision[1], 'r', label='catheter')
    plt.plot(toolRecall[2], toolPrecision[2], 'g', label='dilator')
    plt.plot(toolRecall[3], toolPrecision[3], 'b', label='guidewire')
    plt.plot(toolRecall[4], toolPrecision[4], 'c', label='guidewire_casing')
    plt.plot(toolRecall[5], toolPrecision[5], 'y', label='scalpel')
    plt.plot(toolRecall[6], toolPrecision[6], 'k', label='syringe')
    plt.title('Precision Recall Curve')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend()
    plt.show()
    meanAP = numpy.mean(numpy.array(classAP))
    print('mAP: {}'.format(meanAP))
    return (meanAP, classAP)

def getEqualTrueAndPred(trueLabels,predLabels):
    trueAndPredLabels = pandas.DataFrame(columns=['true','pred','x_min_true','y_min_true','x_max_true','y_max_true','x_min_pred','y_min_pred','x_max_pred','y_max_pred','a','c','d','g','gc','sc','sy','bg'])
    checkedImages = []
    for i in range(len(predLabels)):
        currentFile = predLabels['filePath'][i]
        if not currentFile in checkedImages:
            checkedImages.append(currentFile)
            groundTruthBoxes = trueLabels.loc[trueLabels['filePath']== currentFile]
            predBoxes = predLabels.loc[predLabels['filePath']==currentFile]
            groundTruthIndices = groundTruthBoxes.index
            predBoxIndices = predBoxes.index
            for j in predBoxIndices:
                Found = False
                for k in groundTruthIndices:
                    if predBoxes['label'][j] == groundTruthBoxes['class_name'][k]:
                        Found = True
                        trueAndPredLabels = trueAndPredLabels.append({'true':groundTruthBoxes['class_name'][k],'pred':predBoxes['label'][j],'x_min_true':groundTruthBoxes['x_min'][k],'y_min_true':groundTruthBoxes['y_min'][k],'x_max_true':groundTruthBoxes['x_max'][k],'y_max_true':groundTruthBoxes['y_max'][k],'x_min_pred':predBoxes['xmin'][j],'y_min_pred':predBoxes['ymin'][j],'x_max_pred':predBoxes['xmax'][j],'y_max_pred':predBoxes['ymax'][j],'a':predBoxes['a'][j],'c':predBoxes['c'][j],'d':predBoxes['d'][j],'g':predBoxes['g'][j],'gc':predBoxes['gc'][j],'sc':predBoxes['sc'][j],'sy':predBoxes['sy'][j],'bg':predBoxes['bg'][j]},ignore_index=True)
                if not Found:
                    trueAndPredLabels = trueAndPredLabels.append(
                        {'true': 'none', 'pred': predBoxes['label'][j],
                         'x_min_true': 0, 'y_min_true': 0,
                         'x_max_true': 0, 'y_max_true': 0,
                         'x_min_pred': predBoxes['xmin'][j], 'y_min_pred': predBoxes['ymin'][j],
                         'x_max_pred': predBoxes['xmax'][j], 'y_max_pred': predBoxes['ymax'][j],
                         'a': predBoxes['a'][j], 'c': predBoxes['c'][j], 'd': predBoxes['d'][j], 'g': predBoxes['g'][j],
                         'gc': predBoxes['gc'][j], 'sc': predBoxes['sc'][j], 'sy': predBoxes['sy'][j],
                         'bg': predBoxes['bg'][j]},ignore_index=True)
            for j in groundTruthIndices:
                Found = False
                for k in predBoxIndices:
                    if predBoxes['label'][k] == groundTruthBoxes['class_name'][j]:
                        Found = True
                if not Found:
                    trueAndPredLabels = trueAndPredLabels.append(
                        {'true': groundTruthBoxes['class_name'][j], 'pred': 'none',
                         'x_min_true': groundTruthBoxes['x_min'][j], 'y_min_true': groundTruthBoxes['y_min'][j],
                         'x_max_true': groundTruthBoxes['x_max'][j], 'y_max_true': groundTruthBoxes['y_max'][j],
                         'x_min_pred': 0, 'y_min_pred': 0,
                         'x_max_pred': 0, 'y_max_pred': 0,
                         'a': 0, 'c': 0, 'd': 0, 'g': 0,
                         'gc': 0, 'sc': 0, 'sy': 0,
                         'bg': 1},ignore_index=True)
    return trueAndPredLabels


def getRankedToolPredictions(trueLabels,predLabels,toolName,toolAcronym):
    samples = pandas.DataFrame(columns=['true','pred','confidence'])
    trueAndPredLabels = getEqualTrueAndPred(trueLabels,predLabels)
    numPositiveSamples = 0
    for i in range(len(trueAndPredLabels)):
        if trueAndPredLabels['true'][i] == trueAndPredLabels['pred'][i] and trueAndPredLabels['true'][i] == toolName:
                numPositiveSamples += 1
                trueBBox = [trueAndPredLabels['x_min_true'][i], trueAndPredLabels['y_min_true'][i],
                            trueAndPredLabels['x_max_true'][i], trueAndPredLabels['y_max_true'][i]]
                predBBox = [trueAndPredLabels['x_min_pred'][i], trueAndPredLabels['y_min_pred'][i], trueAndPredLabels['x_max_pred'][i],
                            trueAndPredLabels['y_max_pred'][i]]
                iou = getIOU(trueBBox, predBBox)
                if iou > 0.5:
                    samples = samples.append({'true':1,'pred':1,'confidence':trueAndPredLabels[toolAcronym][i]},ignore_index=True)
                else:
                    samples = samples.append({'true':1,'pred':0, 'condfidence': trueAndPredLabels[toolAcronym][i]}, ignore_index=True)
        elif trueAndPredLabels['true'][i] != trueAndPredLabels['pred'][i] and trueAndPredLabels['pred'][i] == toolName:
                numPositiveSamples += 1
                samples = samples.append({'true':0,'pred':1, 'condfidence': trueAndPredLabels[toolAcronym][i]}, ignore_index=True)
        elif trueAndPredLabels['true'][i] != trueAndPredLabels['pred'][i] and trueAndPredLabels['true'][i] == toolName:
            samples = samples.append({'true': 1, 'pred': 0, 'condfidence': trueAndPredLabels[toolAcronym][i]},
                                     ignore_index=True)
        else:
            samples = samples.append({'true':0,'pred':0, 'condfidence': trueAndPredLabels[toolAcronym][i]}, ignore_index=True)
    samples = samples.sort_values(by=['confidence'],ascending=False)
    predRowIndexes = samples.index
    rankedSamples = pandas.DataFrame(columns=['true','pred','confidence'])
    for i in predRowIndexes:
        rankedSamples = rankedSamples.append({'true':samples['true'][i],'pred':samples['pred'][i],'confidence':samples['confidence'][i]}, ignore_index=True)
    return (rankedSamples,numPositiveSamples)

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
                #match bboxes up based on iou
                #if bboxes don't meet minimum iou requirements, then label as none
                for j in predRowIndexes:
                    Plabel = allPredBBoxes['label'][j]
                    Tlabel = 'none'
                    predbbox = [allPredBBoxes['x_min'][j],allPredBBoxes['y_min'][j],allPredBBoxes['x_max'][j],allPredBBoxes['y_max'][j]]
                    bestIOU = 0
                    for k in trueRowIndexes:
                        truebbox = [allTrueBBoxes['x_min'][k], allTrueBBoxes['y_min'][k], allTrueBBoxes['x_max'][k],allTrueBBoxes['y_max'][k]]
                        iou = getIOU(predbbox,truebbox)
                        if iou > bestIOU:
                            pass
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
        tool = 'anesthetic'
        getMAP(trueLabels,predictedLabels)
        #(accuracy,totalFrameCount) = getAccuracy(trueLabels,predictedLabels)
        #print (accuracy)
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



