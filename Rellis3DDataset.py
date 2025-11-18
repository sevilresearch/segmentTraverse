import os
from typing import Callable, Optional
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image


class Rellis3D(Dataset):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.imagesList = []
        self.annotationsList = []

        if (split != "train") and (split != "val") and (split != "test"):
            print("Error: split must be 'train', 'val', or 'test'")
            exit(0)

        self.imagesPath = root + "Images/Images/" + split + "/"
        self.annotationsPath = root + "Images/Annotations/" + split + "/"

        imagesList = []
        annotationsList = []

        for imagePath in os.listdir(self.imagesPath):
            imagesList.append(imagePath)

        for annotationPath in os.listdir(self.annotationsPath):
            annotationsList.append(annotationPath)

        self.imagesList = imagesList
        self.annotationsList = annotationsList

    def __getitem__(self, index):
        image = Image.open(self.imagesPath + self.imagesList[index]).convert('RGB')
        target = Image.open(self.annotationsPath + self.annotationsList[index])

        #Remapping labels
        label_mapping = {0: 0,
                        1: 0,
                        3: 1,
                        4: 2,
                        5: 3,
                        6: 4,
                        7: 5,
                        8: 6,
                        9: 7,
                        10: 8,
                        12: 9,
                        15: 10,
                        17: 11,
                        18: 12,
                        19: 13,
                        23: 14,
                        27: 15,
                        29: 1,
                        30: 1,
                        31: 16,
                        32: 4,
                        33: 17,
                        34: 18}

        target = np.array(target)

        remappedTarget = target.copy()

        for key, value in label_mapping.items():
            remappedTarget[target == key] = value

        remappedTarget = Image.fromarray(remappedTarget)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            remappedTarget = self.target_transform(remappedTarget)

        return (image, remappedTarget)

    def __len__(self):
        return len(self.imagesList)