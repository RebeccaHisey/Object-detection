import os
import csv
import ast
import math

def readCSV(filePath):
    fileRows = []
    with open(filePath) as csvFile:
        csvReader = csv.reader(csvFile)
        for entry in csvReader:
            fileRows.append(entry)
    return fileRows

def getBoundingBoxes(fileRows):
    boundingBoxes = []
    for i in range(1,len(fileRows)):
        fileName = fileRows[i][0]
        boundingBoxDict = ast.literal_eval(fileRows[i][5])
        if boundingBoxDict != {}:
            x,y,width,height = boundingBoxDict["x"],boundingBoxDict["y"],boundingBoxDict["width"],boundingBoxDict["height"]
        else:
            x,y,width,height = math.inf,math.inf,math.inf,math.inf
        boundingBoxes.append([fileName,x,y,x+width,y+height])
    return boundingBoxes

def writeCSV(filePath,boundingBoxes):
    with open(filePath,"w") as csvFile:
        csvWriter = csv.writer(csvFile)
        for row in boundingBoxes:
            csvWriter.writerow(row)

def main():
    filePath = 'd:/Pilot_Study/'
    videoIDs = os.listdir(os.path.join(filePath,'Segmented_videos'))
    for video in videoIDs:
        print(video)
        videoPath = os.path.join(filePath,video+'.csv')
        adaptiveFilePath = os.path.join(filePath,video + '_adapted.csv')
        fileRows = readCSV(videoPath)
        boundingBoxes = getBoundingBoxes(fileRows)
        writeCSV(adaptiveFilePath,boundingBoxes)
main()
