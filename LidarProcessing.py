import os
import math
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt

class LidarProcessor():

    def __init__(self, datasetPath, imageSize, imageResize):
        self.imageHeight = imageSize[0]
        self.imageWidth = imageSize[1]

        self.resizeHeight = imageResize[0]
        self.resizeWidth = imageResize[1]

        self.resizeRatio = (imageResize[0] / imageSize[0], imageResize[1] / imageSize[1])

        self.cameraMatrix = np.array([[2813.643275, 0          , 969.285772],
                                      [0          , 2808.326079, 624.049972],
                                      [0          , 0          , 1         ]])

        self.transformationMatrices = []
        self.rotationVectors = []
        self.translationVectors = []

        self.getTransforms(datasetPath)

        self.distortionCoefficients = np.array([-0.134313, -0.025905, 0.002181, 0.00084, 0]).reshape(5, 1)

        self.FOVX = 2 * np.arctan2(self.imageWidth, 2 * self.cameraMatrix[0, 0]) * 180 / np.pi + 10
        self.FOVY = 2 * np.arctan2(self.imageHeight, 2 * self.cameraMatrix[1, 1]) * 180 / np.pi + 10

        self.redGreenGradient = []

        for green in range(0, 255, 2):
            self.redGreenGradient.append([255, green, 0])

        for red in range(255, 0, -2):
            self.redGreenGradient.append([red, 255, 0])

        #Percentage of point distance range, if a point is past this range, local traversability is removed
        self.cutoffRange = 0.1
        self.cutoffIndex = len(self.redGreenGradient) * self.cutoffRange


    def getTransforms(self, datasetPath):
        transformsPath = datasetPath + "CameraToLidarTransforms/"

        for file in os.listdir(transformsPath):
            self.transformationMatrices.append(self.getLidarToCameraMatrix(transformsPath + file))

        for transformationMatrix in self.transformationMatrices:
            rotationVector, translationVector = self.extractRotationAndTranslationVecs(transformationMatrix)
            self.rotationVectors.append(rotationVector)
            self.translationVectors.append(translationVector)

    def extractRotationAndTranslationVecs(self, transformationMatrix):
        rotationTransformMatrix = transformationMatrix[:3, :3]
        translationTransformVector = transformationMatrix[:3, 3]

        translationTransformVector = translationTransformVector.reshape(3, 1)

        rotationVector, _ = cv2.Rodrigues(rotationTransformMatrix)
        translationVector = translationTransformVector

        return rotationVector, translationVector


    def placePointsOnImage(self, points, colors, image):
        #Color image
        if len(image.shape) == 3:
            for i in range(points.shape[1]):
                projectedX, projectedY = np.int32(points[0][i] * self.resizeRatio[1]), np.int32(points[1][i] * self.resizeRatio[0])

                if (projectedX > 0) and (projectedX < self.resizeWidth) and (projectedY > 0) and (projectedY < self.resizeHeight):
                    image[projectedY][projectedX] = colors[i]

        #Grayscale image
        else:
            for i in range(points.shape[1]):
                projectedX, projectedY = np.int32(points[0][i] * self.resizeRatio[1]), np.int32(points[1][i] * self.resizeRatio[0])

                if (projectedX > 0) and (projectedX < self.resizeWidth) and (projectedY > 0) and (projectedY < self.resizeHeight):
                    image[projectedY][projectedX] = 255

        return image

    def removeCutoffTraversability(self, points, colors, image):
        lidarCoverageMap = np.zeros(image.shape)

        for i in range(points.shape[1]):
            projectedX, projectedY = np.int32(points[0][i] * self.resizeRatio[1]), np.int32(points[1][i] * self.resizeRatio[0])

            if (projectedX > 0) and (projectedX < self.resizeWidth) and (projectedY > 0) and (projectedY < self.resizeHeight):
                #Build LiDAR coverage map
                cv2.circle(lidarCoverageMap, (projectedX, projectedY), 10, 1, cv2.FILLED)

                #Remove distant LiDAR regions
                if colors[i] == 0:
                    cv2.circle(image, (projectedX, projectedY), 7, 0, cv2.FILLED)

        #Remove regions where LiDAR is not present
        image = np.logical_and(image, lidarCoverageMap)
        image = image.astype(np.uint8)

        return image

    def calculateAndPlaceRegionCenters(self, image):
        numLabels, labels, stats, centerPoints = cv2.connectedComponentsWithStats(image, connectivity=4)

        for i in range(1, numLabels):
            if stats[i, cv2.CC_STAT_AREA] > 20000:
                centerX, centerY = centerPoints[i]
                centerX, centerY = math.floor(centerX), math.floor(centerY)

                cv2.line(image, (centerX - 10, centerY), (centerX + 10, centerY), 2, 1)
                cv2.line(image, (centerX, centerY - 10), (centerX, centerY + 10), 2, 1)

        return image

    def removeTraversableIslands(self, image, numLabels, stats):
        for i in range(1, numLabels):
            leftBound = stats[i, cv2.CC_STAT_LEFT]
            topBound = stats[i, cv2.CC_STAT_TOP]
            rightBound = leftBound + stats[i, cv2.CC_STAT_WIDTH]
            bottomBound = topBound + stats[i, cv2.CC_STAT_HEIGHT]

            #Calculate
            if stats[i, cv2.CC_STAT_AREA] < 20000:
                cv2.rectangle(image, (leftBound, topBound), (rightBound, bottomBound), 0, -1)

        return image


    def calculateColors(self, pointDistances):
        minDist = pointDistances.min()
        maxDist = pointDistances.max()

        colorIndexs = (((pointDistances - minDist) * (255 - 0)) / (maxDist - minDist)) + 0
        colorIndexs = np.uint8(colorIndexs)

        colors = []

        for colorIndex in colorIndexs:
            colors.append(self.redGreenGradient[colorIndex])

        return colors

    def calculateCutoffPoints(self, pointDistances):
        minDist = pointDistances.min()
        maxDist = pointDistances.max()

        colorIndexs = (((pointDistances - minDist) * (255 - 0)) / (maxDist - minDist)) + 0
        colorIndexs = np.uint8(colorIndexs)

        colors = []

        for colorIndex in colorIndexs:
            if colorIndex > self.cutoffIndex:
                colors.append(0)
            else:
                colors.append(1)

        return colors

    def filterPoints(self, points, transformationMatrix, traversabilityMode):
        transformationMatrix = np.array(transformationMatrix)

        p_l = np.ones((points.shape[0], points.shape[1] + 1))
        p_l[:, :3] = points
        p_c = np.matmul(transformationMatrix, p_l.T)
        p_c = p_c.T
        x = p_c[:, 0]
        y = p_c[:, 1]
        z = p_c[:, 2]

        xangle = np.arctan2(x, z) * 180 / np.pi
        yangle = np.arctan2(y, z) * 180 / np.pi

        flag2 = (xangle > -self.FOVX / 2) & (xangle < self.FOVX / 2)
        flag3 = (yangle > -self.FOVY / 2) & (yangle < self.FOVY / 2)

        res = p_l[flag2 & flag3, :3]
        res = np.array(res)

        x = res[:, 0]
        y = res[:, 1]
        z = res[:, 2]
        dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        if traversabilityMode:
            colors = self.calculateCutoffPoints(dist)

        else:
            colors = self.calculateColors(dist)

        return res, colors

    def getLidarToCameraMatrix(self, filePath):
        with open(filePath, "r") as file:
            transforms = yaml.load(file, Loader=yaml.Loader)

        rotationTransform = transforms['os1_cloud_node-pylon_camera_node']['q']
        rotationTransform = np.array([rotationTransform['x'], rotationTransform['y'], rotationTransform['z'], rotationTransform['w']])
        translationTransform = transforms['os1_cloud_node-pylon_camera_node']['t']
        translationTransform = np.array([translationTransform['x'], translationTransform['y'], translationTransform['z']])
        rotationMatrix = Rotation.from_quat(rotationTransform).as_matrix()

        inverseTransformationMatrix = np.eye(4, 4)
        inverseTransformationMatrix[:3, :3] = rotationMatrix
        inverseTransformationMatrix[:3, -1] = translationTransform
        transformationMatrix = np.linalg.inv(inverseTransformationMatrix)

        return transformationMatrix

    def projectPointsToImage(self, image, pointCloud, transformType):
        transformationMatrix = self.transformationMatrices[transformType]
        rotationVector = self.rotationVectors[transformType]
        translationVector = self.translationVectors[transformType]

        filteredPoints, pointColors = self.filterPoints(pointCloud, transformationMatrix, False)

        projectedPoints, _ = cv2.projectPoints(filteredPoints, rotationVector, translationVector, self.cameraMatrix, self.distortionCoefficients)

        projectedPoints = np.squeeze(projectedPoints, 1)
        projectedPoints = projectedPoints.T

        imageLidarFusion = self.placePointsOnImage(projectedPoints, pointColors, image)

        return imageLidarFusion

    def calculateTraversability(self, image, pointCloud, transformType):
        transformationMatrix = self.transformationMatrices[transformType]
        rotationVector = self.rotationVectors[transformType]
        translationVector = self.translationVectors[transformType]

        filteredPoints, pointColors = self.filterPoints(pointCloud, transformationMatrix, True)

        projectedPoints, _ = cv2.projectPoints(filteredPoints, rotationVector, translationVector, self.cameraMatrix, self.distortionCoefficients)

        projectedPoints = np.squeeze(projectedPoints, 1)
        projectedPoints = projectedPoints.T

        return image