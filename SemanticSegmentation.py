import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torchvision import transforms
from torchvision import datasets
from Rellis3DDataset import Rellis3D
import matplotlib.pyplot as plt


#dataset = "Cityscapes"
dataset = "Rellis3D"

modelSavesPath = "C:/Users/maste/Desktop/PyTorchSegmentation/ModelSaves/"
segmentationsPath = "C:/Users/maste/Desktop/PyTorchSegmentation/Segmentations/"
imageResize = (256, 512)

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

if dataset == "Cityscapes":
    datasetPath = "C:/Users/maste/Desktop/PyTorchSegmentation/Datasets/Cityscapes/"
    numClasses = 34
    testDataset = datasets.Cityscapes(datasetPath, mode="fine", split="val", target_type="semantic", transform=normalizeTransform, target_transform=resizeTransform)

elif dataset == "Rellis3D":
    datasetPath = "C:/Users/maste/Desktop/PyTorchSegmentation/Datasets/Rellis3D/Images/"
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
cumulativeAccuracy = 0
numImages = len(testDataLoader)
imageSize = imageResize[0] * imageResize[1]
classIntersectionCounts = torch.zeros(numClasses)
classUnionCounts = torch.zeros(numClasses)

for testBatch, targetBatch in testDataLoader:
    if (imagesTested % 20) == 0:
        print("Processing image " + str(imagesTested) + " out of " + str(numImages) + ".")

    testBatch = testBatch.to(device)

    with torch.no_grad():
        outputBatch = segmentationModel(testBatch)["out"]#.to("cpu")

    outputBatchPredictions = outputBatch.argmax(1)

    targetBatch = (targetBatch.squeeze(1) * 255).type(torch.LongTensor)

    testImage, targetImage, outputImage = testBatch[0].to("cpu"), targetBatch[0].to("cpu"), outputBatchPredictions[0].to("cpu")

    #PIXEL ACCURACY
    correctPixels = torch.count_nonzero(torch.eq(targetImage, outputImage)).item()
    pixelAccuracy = correctPixels / imageSize

    cumulativeAccuracy += pixelAccuracy

    plt.imshow(testImage.permute(1, 2, 0))
    plt.show()

    plt.imshow(targetImage)
    plt.show()

    plt.imshow(outputImage)
    plt.show()

    #IOUS
    for classIndex in range(numClasses):
        targetClassMask = (classIndex == targetImage)
        predictedClassMask = (classIndex == outputImage)

        """
        plt.imshow(testImage.permute(1, 2, 0))
        plt.show()

        plt.imshow(targetImage)
        plt.show()

        plt.imshow(outputImage)
        plt.show()

        plt.imshow(targetClassMask)
        plt.show()

        plt.imshow(predictedClassMask)
        plt.show()
        """

        classIntersection = torch.logical_and(targetClassMask, predictedClassMask)
        classUnion = torch.logical_or(targetClassMask, predictedClassMask)

        # plt.imshow(classIntersection)
        # plt.show()

        #plt.imshow(classUnion)
        #plt.show()

        classIntersectionCounts[classIndex] += torch.count_nonzero(classIntersection).item()
        classUnionCounts[classIndex] += torch.count_nonzero(classUnion).item()

    imagesTested += 1
    
    """
    # step1: morphological filtering (helps splitting parts that don't belong to the person blob)
    kernel = np.ones((13, 13), np.uint8)  # hardcoded 13 simply gave nice results
    segmentedImageMF = cv.morphologyEx(np.array(segmentedImage), cv.MORPH_OPEN, kernel)
    """

overallPixelAccuracy = (cumulativeAccuracy / numImages) * 100
classIoUs = (classIntersectionCounts / classUnionCounts).tolist()
overallIoU = (torch.sum(classIntersectionCounts) / torch.sum(classUnionCounts)).item()

print("Overall Pixel Accuracy: " + str(overallPixelAccuracy))
print("Class IoUs: " + str(classIoUs))
print("Overall IoU: " + str(overallIoU))