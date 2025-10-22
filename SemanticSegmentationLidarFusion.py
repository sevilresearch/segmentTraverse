import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torchvision import transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from Rellis3DDatasetWithLidar import Rellis3D
from LidarProcessing import LidarProcessor
from PathingProcessing import PathingProcessor

dataset = "Rellis3D"

modelSavesPath = "C:/Python/PyTorchSegmentation/ModelSaves/"
segmentationsPath = "C:/Python/PyTorchSegmentation/Segmentations/"
imageSize = (1200, 1920)
imageResize = (512, 1024)
#imageResize = (256, 512)

#pathingType = "StraightLine"
#pathingType = "MyAlg"
pathingType = "AStar"
#pathingType = "MaxSafe"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")

#TODO Fix normalization
resizeTransform = transforms.Compose([
        transforms.Resize(imageResize),
        transforms.ToTensor(),
])

normalizeTransform = transforms.Compose([
        transforms.Resize(imageResize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Dataset definition
datasetPath = ""
numClasses = 0
testDataset = None

if dataset == "Rellis3D":
    datasetPath = "C:/Python/PyTorchSegmentation/Datasets/Rellis3D/"
    numClasses = 19
    testDataset = Rellis3D(datasetPath, split="val", transform=normalizeTransform, target_transform=resizeTransform)

else:
    print("Error: Please define a valid dataset")
    exit(0)

testSampler = torch.utils.data.SequentialSampler(testDataset)
testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=1, sampler=testSampler)

segmentationModel = models.segmentation.deeplabv3_resnet101(pretrained=True)
segmentationModel.classifier = DeepLabHead(2048, numClasses)
segmentationModel.load_state_dict(torch.load(modelSavesPath + "DeeplabV3" + dataset + "-6-0.7544873356819153.pth"))
segmentationModel.eval()
segmentationModel.to(device)

imagesTested = 0
numImages = len(testDataLoader)

cumulativePathingCalculationTime = 0

totalPathLength = 0
totalNumPaths = 0
totalUnsafePathPixels = 0

lidarProcessor = LidarProcessor(datasetPath, imageSize, imageResize)
pathingProcessor = PathingProcessor()

classColorLookupTable = [[108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153],
                         [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64],
                         [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204],
                         [170, 170, 170], [41, 121, 255], [134, 255, 239], [99, 66, 34], [110, 22, 138]]

nontraversable = [0, 0, 0]
traversable = [0, 0, 255]
#sky = [128, 255, 255]
#obstacles = [128, 128, 128]

traversabilityLookupTable = [traversable, traversable, nontraversable, nontraversable,
                             nontraversable, nontraversable, nontraversable, nontraversable, traversable,
                             nontraversable, nontraversable, nontraversable, nontraversable, nontraversable,
                             traversable, nontraversable, traversable, traversable, nontraversable]

"""
traversabilityLookupTable = [traversable, traversable, obstacles, obstacles,
                             nontraversable, sky, obstacles, obstacles, traversable,
                             obstacles, obstacles, obstacles, obstacles, nontraversable,
                             traversable, obstacles, traversable, traversable, obstacles]
"""

#table = np.array([((i / 255.0)) * 255 for i in np.arange(0, 256)]).astype("uint8")

identity = np.arange(256, dtype = np.dtype('uint8'))
zeros = np.zeros(256, np.dtype('uint8'))
lut = np.dstack((identity, identity, zeros))

for i in range(256 - len(classColorLookupTable)):
    classColorLookupTable.append([0, 0, 0])
    traversabilityLookupTable.append([0, 0, 0])

overallStart = time.time()

for index, testBatch, targetBatch, pointCloud, transformType in testDataLoader:
    if (imagesTested % 20) == 0:
        print("Processing image " + str(imagesTested) + " out of " + str(numImages) + ".")

    testBatch = testBatch.to(device)

    with torch.no_grad():
        outputBatch = segmentationModel(testBatch)["out"]

    outputBatchPredictions = outputBatch.argmax(1)

    targetBatch = (targetBatch.squeeze(1) * 255).type(torch.LongTensor)

    testImage, targetImage, outputImage = testBatch[0].to("cpu"), targetBatch[0].to("cpu"), outputBatchPredictions[0].to("cpu")

    outputImage = cv2.cvtColor(np.uint8(np.asarray(outputImage)), cv2.COLOR_GRAY2RGB)
    traversabilityImage = outputImage

    outputImage = cv2.LUT(outputImage, np.array([classColorLookupTable]))
    traversabilityImage = cv2.LUT(traversabilityImage, np.array([traversabilityLookupTable]))

    if index in [0, 126, 177, 333, 715, 821, 982]:
        plt.imsave("tempExamples4/Seg" + str(imagesTested) + "-1.png", np.uint8(outputImage))

    traversabilityImage = cv2.cvtColor(np.uint8(traversabilityImage), cv2.COLOR_RGB2GRAY)
    traversabilityImage = cv2.threshold(traversabilityImage, 1, 1, cv2.THRESH_BINARY)[1]

    if index in [0, 126, 177, 333, 715, 821, 982]:
        plt.imsave("tempExamples4/Seg" + str(imagesTested) + "-T.png", np.uint8(traversabilityImage))

    print(imagesTested)

    #outputWithProjectedPoints = lidarProcessor.projectPointsToImage(np.asarray(outputImage), pointCloud.numpy()[0, :, :], int(transformType[0]))
    #traversabilityImage = lidarProcessor.calculateTraversability(np.asarray(traversabilityImage), pointCloud.numpy()[0, :, :], int(transformType[0]))

    pathingAreaImage = pathingProcessor.calculatePathingAreaFromTraversableArea(traversabilityImage)
    combinedTraversabilityAndPathingAreaImage = traversabilityImage + pathingAreaImage

    if index in [0, 126, 177, 333, 715, 821, 982]:
        plt.imsave("tempExamples4/Seg" + str(imagesTested) + "-2.png", np.uint8(combinedTraversabilityAndPathingAreaImage))

    if pathingType == "StraightLine":
        numLabels, labels, stats, centerPoints = cv2.connectedComponentsWithStats(pathingAreaImage, connectivity=4)

        pathingStart = time.time()
        pathingImage, numPaths, pathLength, unsafePathPixels = pathingProcessor.straightLinePathing(traversabilityImage, combinedTraversabilityAndPathingAreaImage, numLabels, stats)
        cumulativePathingCalculationTime += time.time() - pathingStart

        totalPathLength += pathLength
        totalNumPaths += numPaths
        totalUnsafePathPixels += unsafePathPixels

        plt.imsave("StraightLinePathingImages/Seg" + str(imagesTested) + "-1.png", np.uint8(outputImage))
        plt.imsave("StraightLinePathingImages/Seg" + str(imagesTested) + "-S.png", np.uint8(pathingImage))

    elif pathingType == "MyAlg":
        numLabels, labels, stats, centerPoints = cv2.connectedComponentsWithStats(pathingAreaImage, connectivity=4)

        pathingStart = time.time()
        pathingImage, numPaths, pathLength, unsafePathPixels = pathingProcessor.MyAlgPathing(combinedTraversabilityAndPathingAreaImage, numLabels, stats)
        cumulativePathingCalculationTime += time.time() - pathingStart

        totalPathLength += pathLength
        totalNumPaths += numPaths
        totalUnsafePathPixels += unsafePathPixels

        plt.imsave("MyAlgPathingImages/Seg" + str(imagesTested) + "-1.png", np.uint8(outputImage))
        plt.imsave("MyAlgPathingImages/Seg" + str(imagesTested) + "-2.png", np.uint8(pathingImage))

    elif pathingType == "AStar":
        numLabels, labels, stats, centerPoints = cv2.connectedComponentsWithStats(pathingAreaImage, connectivity=4)

        pathingStart = time.time()
        pathingImage, numPaths, pathLength = pathingProcessor.AStarPathing(combinedTraversabilityAndPathingAreaImage, numLabels, stats)
        cumulativePathingCalculationTime += time.time() - pathingStart

        totalPathLength += pathLength
        totalNumPaths += numPaths

        plt.imsave("AStarPathingImages/Seg" + str(imagesTested) + "-1.png", np.uint8(outputImage))
        plt.imsave("AStarPathingImages/Seg" + str(imagesTested) + "-3.png", np.uint8(pathingImage))

        if index in [0, 126, 177, 333, 715, 821, 982]:
            plt.imsave("tempExamples4/Seg" + str(imagesTested) + "-3.png", np.uint8(pathingImage))

    elif pathingType == "MaxSafe":
        numLabels, labels, stats, centerPoints = cv2.connectedComponentsWithStats(traversabilityImage, connectivity=4)

        pathingStart = time.time()
        pathingImage, numPaths, pathLength = pathingProcessor.MaxSafePathing(traversabilityImage, numLabels, stats)
        cumulativePathingCalculationTime += time.time() - pathingStart

        totalPathLength += pathLength
        totalNumPaths += numPaths

        plt.imsave("MaxSafePathingImages/Seg" + str(imagesTested) + "-1.png", np.uint8(outputImage))
        plt.imsave("MaxSafePathingImages/Seg" + str(imagesTested) + "-4.png", np.uint8(pathingImage))

    imagesTested += 1

overallTime = time.time() - overallStart

print("Average path length is " + str(totalPathLength / totalNumPaths) + " for " + pathingType + ".")
print(str((totalUnsafePathPixels / totalPathLength) * 100) + "% of paths was unsafe.")
print("Total runtime was " + str(overallTime) + " seconds.")
print("Average runtime per image was " + str(overallTime / (imagesTested + 1)) + " seconds.")
print("Total pathing runtime was " + str(cumulativePathingCalculationTime) + " seconds.")
print("Average pathing runtime per image was " + str(cumulativePathingCalculationTime / (imagesTested + 1)) + " seconds.")