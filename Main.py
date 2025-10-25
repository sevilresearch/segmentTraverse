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
import floodfill
from PathfindingAndImageProcessing import PathingProcessor

dataset = "Rellis3D"
modelSavesPath = "C:/Python/PyTorchSegmentation/ModelSaves/"
segmentationsPath = "C:/Python/PyTorchSegmentation/Segmentations/"
imageSize = (1200, 1920)
imageResize = (640, 1024)
# imageResize = (256, 512)

# pathingType = "StraightLine"
# pathingType = "MyAlg"
pathingType = "AStar"
# pathingType = "MaxSafe"

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cpu")

resizeTransform = transforms.Compose([
    transforms.Resize(imageResize),
    transforms.ToTensor(),
])

normalizeTransform = transforms.Compose([
    transforms.Resize(imageResize),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# Dataset definition
datasetPath = ""
numClasses = 0
testDataset = None

if dataset == "Rellis3D":
    datasetPath = "C:/Python/PyTorchSegmentation/Rellis3D/"
    numClasses = 19
    testDataset = Rellis3D(datasetPath, split = "val",
                           transform=normalizeTransform, target_transform=resizeTransform)
else:
    print("Error: Please define a valid dataset")
    exit(0)

testSampler = torch.utils.data.SequentialSampler(testDataset)
testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=1, sampler=testSampler)

segmentationModel = models.segmentation.deeplabv3_resnet101(pretrained=True)
segmentationModel.classifier = DeepLabHead(2048, numClasses)
segmentationModel.load_state_dict(
    torch.load(modelSavesPath + "DeeplabV3" + dataset + "-6-0.7544873356819153.pth", map_location=torch.device('cpu'))
)
segmentationModel.eval()
segmentationModel.to(device)

kvals = [40]
# kvals = [5, 25, 50, 55, 100, 150, 200]

element = "Circle"
operation = "Open"
rundescriptor_abrv = element + operation
# rundescriptor = "Morphological Opening w/ " + element + "Structuring Element and Imfill"

for runksize in kvals:
    imagesTested = 0
    numImages = len(testDataLoader)
    cumulativePathingCalculationTime = 0
    totalPathLength = 0
    totalNumPaths = 0
    totalUnsafePathPixels = 0

    # lidarProcessor = LidarProcessor(datasetPath, imageSize, imageResize)
    pathingProcessor = PathingProcessor()

    # This is a definition of the colors for each segmentation class (19)
    classColorLookupTable = [
        [108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153],
        [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64],
        [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204],
        [170, 170, 170], [41, 121, 255], [134, 255, 239], [99, 66, 34], [110, 22, 138]]

    nontraversable = [0, 0, 0]
    traversable = [0, 0, 255]
    # sky = [128, 255, 255]
    # obstacles = [128, 128, 128]

    # These are the traversability classifications for each segmentation class
    traversabilityLookupTable = [
        traversable, traversable, nontraversable, nontraversable,
        nontraversable, nontraversable, nontraversable, nontraversable, traversable,
        nontraversable, nontraversable, nontraversable, nontraversable, nontraversable,
        traversable, nontraversable, traversable, traversable, nontraversable]

    # table = np.array([((i / 255.0)) * 255 fir i in np.arange(0, 256)]).astype("uint8")

    identity = np.arange(256, dtype=np.dtype('uint8'))
    zeros = np.zeros(256, np.dtype('uint8'))
    lut = np.dstack((identity, identity, zeros))

    for i in range(256 - len(classColorLookupTable)):
        classColorLookupTable.append([0, 0, 0])
        traversabilityLookupTable.append([0, 0, 0])

    overallStart = time.time()

    for index, testBatch, targetBatch, pointCloud, transformType in testDataLoader:
        individualStart = time.time()
        if(imagesTested % 20) == 0:
            print("Processing image " + str(imagesTested) + " out of " + str(numImages) + ".")

        testBatch = testBatch.to(device)

        with torch.no_grad():
            outputBatch = segmentationModel(testBatch)["out"]

        outputBatchPredictions = outputBatch.argmax(1)
        targetBatch = (targetBatch.squeeze(1) * 255).type(torch.LongTensor)
        testImage, targetImage, outputImage = (testBatch[0].to("cpu"), targetBatch[0].to("cpu"),
                                               outputBatchPredictions[0].to("cpu"))
        # plt.imsave(segmentationsPath + tempExamples4/Seg" + str(imagesTested) + "-PRE1.png", np.uint8(outputImage))

        # imfill (before morphological operations)
        # outputImage = floodfill.from_edges(outputImage, four_way=True)
        #plt.imsave(segmentationsPath + "tempExamples4/Seg" + str(imagesTested) + "-imfill" + " k" + str(runksize) +
        # str(rundescriptor_abrv) + ".png" np.uint8(outputImage))

        #grayscale ->RGB
        outputImage = cv2.cvtColor(np.uint8(np.asarray(outputImage)), cv2.COLOR_GRAY2RGB)
        plt.imsave(segmentationsPath + "tempExamples4/Seg" + str(imagesTested) + "-PRE2.png", np.uint8(outputImage))

        # begin post-Rellis Segmentation Morphology Operations
        # morph open
        # outputImage = cv2.morphologyEx(outputImage, cv2.MORPH_OPEN, np.ones((30,30), np.uint8))
        # outputImage = cv2.morphologyEx(outputImage, cv2.MORPH_OPEN, np.ones((runksize, runksize), np.uint8))
        # outputImage = cv2.morphologyEx(outputImage, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (runksize, runksize)))
        #plt.imsave(segmentationsPath + "tempExamples4/Seg" + str(imagesTested) + "open k" + str(runksize) + str(rundescriptor_abrv) + ".png", np.uint8(outputImage))

        #morph close
        # outputImage = cv2.morphologyEx(outputImage, cv2.MORPH_CLOSE, np.ones((30,30), np.uint8))
        # outputImage = cv2.morphologyEx(outputImage, cv2.MORPH_CLOSE, np.ones((runksize, runksize), np.uint8))
        # outputImage = cv2.morphologyEx(outputImage, cv2.MORPH_CLOSE, cv2.getStructureElement(cv2.MORPH_ELLIPSE, (runksize, runksize)))
        # plt.imsave(segmentationPath + "tempexamples4/Seg" + str(imagesTested) + "close k" + str(runksize) + str(rundescriptor+abrv) + ".png", np.uint8(outputImage))

        # imfill (after morphological operations)
        # outputImage = cv2.cvtColor(np.uint8(np.asarray(outputImage)), cv2.COLOR_GRAY2RGB)
        # plt.imsave(segmentationPath + "tempExamples4/Seg" + str(imagesTested) + "-imfill" + " k" + str(runksize) + str(rundescriptor_abrv) + ".png", np.uint8(outputImage))

        # erosion
        # outputImage = cv2.erode(outputImage, np.ones((runksize, runksize), np.uint8))
        # outputimage = cv2.erode(outputImage, cv2.getStructureElement(cv2.MORPH_ELLIPSE, (runksize, runksize)), np.uint8))
        # plt.imsave(segmentationsPath + "tempExamples4/Seg" + str(imagesTested) + "MorphE.png", np.uint(outputImage))
        # end erosion

        # end Post-Rellis Segmentation Morphology Operations

        traversabilityImage = outputImage
        outputImage = cv2.LUT(outputImage, np.array([classColorLookupTable]))
        plt.imsave(segmentationsPath + "tempExamples4/Seg" + str(imagesTested) + " k" + str(runksize) + "-PRE3.png", np.uint8(outputImage))

        traversabilityImage = cv2.LUT(traversabilityImage, np.array([traversabilityLookupTable]))
        #plt.imsave(segmentationsPath + "tempExamples4/Seg" + str(imagesTested) + " k" + str(runksize) + "-PRE4.png", np.uint8(traversabilityImage))
        traversabilityImage = cv2.cvtColor(np.uint8(traversabilityImage), cv2.COLOR_RGB2GRAY)
        # plt.imsave(segmentationsPath+ "tempExamples4/Seg" + str(imagesTested) + " k" + str(runksize) + "-G.png", np.uint8(traversabilityImage))
        traversabilityImage = cv2.threshold(traversabilityImage, 1, 1, cv2.THRESH_BINARY)[1]
        plt.imsave(segmentationsPath + "tempExamples4/Seg" + str(imagesTested) + "k" + str(runksize) + "-T.png", np.uint8(traversabilityImage))

        print(imagesTested)

        # experimental lidar
        # outputWithProjectedPoints = lidarProcessor.projectPointsToImage(np.asarray(outputImage), pointCloud.numpy()[0, :, :], int(transformType[0]))
        # traversabilityImage = lidarProcessor.calculateTraversability(np.asarray(traversabilityImage), pointCloud.numpy()[0, :, :], int(transformType[0]))

        pathingAreaImage = pathingProcessor.calculatePathingAreaFromTraversableArea(traversabilityImage)
        combinedTraversabilityAndPathingAreaImage = traversabilityImage + pathingAreaImage

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
            individualPathTime = time.time() - pathingStart
            individualTime = time.time() - individualStart
            imagesTestedName = testDataset.imagesList[imagesTested]
            print(imagesTestedName)

            cumulativePathingCalculationTime += time.time() - pathingStart
            totalPathLength += pathLength
            totalNumPaths += numPaths
            with open(('C:/Python/PyTorchSegmentation/TAData/' + str(rundescriptor_abrv) + '.txt'), 'a') as DataTA:
                DataTA.writelines(','.join([str(imagesTestedName), str(pathLength), str(individualTime), str(individualPathTime)]))
                #DataTA.writelines('.join([operation, element, str(runksize), str(AvgLength), str(overallTime), tr(cumulativePathingCalculationTime)]))
                DataTA.write('\n')

            plt.imsave(segmentationsPath + "AStarPathingImages/Seg_" + str(imagesTestedName) + "_" + str(imagesTested) + "k" + str(runksize) + "-out1.png", np.uint8(outputImage))
            plt.imsave(segmentationsPath + "AStarPathingImages/Seg_" + str(imagesTestedName) + "_" + str(imagesTested) + "k" + str(runksize) + "-out3.png", np.int8(pathingImage))

            # if index in [0, 20, 40, 60, 80, 100, 120]:
            # plt.imsave(segmentationsPath + "tempExamples4/Seg" + str(imagesTested) + "-3.png", np.uint8(pathingImage))

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
        #print(str(rundescriptor) + "w/" str(runksize) + "x" + str(runksize) + "Kernel")
        if totalNumPaths != 0:
            AvgLength = totalPathLength/totalNumPaths
        else:
            AvgLength = "totalNumPaths = 0"
        # if totalPathLength != 0:
            # print(str((totalUnsafePathPixels / totalPathLength) * 100) + "% of paths was unsafe.")
        # else:
            # print("totalPathLength = 0")

        averageTime = overallTime / (imagesTested + 1)
        averagePathTime = cumulativePathingCalculationTime / (imagesTested + 1)

        with open(("C:/Python/PyTorchSegmentation/TAData/" + str(rundescriptor_abrv) + '.txt'), 'a') as DataTA:
            DataTA.writelines(','.join([str(averageTime), str(averagePathTime), str(AvgLength), str(overallTime), str(cumulativePathingCalculationTime)]))
            #DataTA.writelines(','.join([operation, element, str(runksize), str(AcgLength), str(overallTime), str(cumulativePathingCalculationTime)]))
            DataTA.write("\n")

        #print("Total runtime was" + str(overallTime) + " Seconds.")
        # # print("Average runtime per image was " + str(overallTime / (imagesTested+ 1)) + " seconds.")
        # print("Total pathing runtime was " + str(cumulativePathingCalculationTime) + " seconds.")
        #print("Average pathing runtime per image was " + str(cumulativePathingCalculationTime / (imagesTested + 1)) + " seconds.")

