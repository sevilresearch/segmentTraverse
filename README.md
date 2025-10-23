This is the repository for the UWF Sevil Research Group's Traversability Assessment of Unstructured Environments Through Image Segmentation and Path Planning. 

Our goal is to segment an image of an outdoor off road location into traversable and non-traversable surfaces. Then, the system will plan a viable path through the area and the platform will execute the path.
The steps to run these files and view the results yourself are listed below. These are subject to change as research progresses.

Version 1.0: 
We used the PyCharm IDE to work on these files, and the steps will reference this IDE.

1. Download Files from GitHub. Also download the Rellis3D dataset we used from this drive link:
   https://drive.google.com/drive/folders/12t654a10328USRkx45VYmUrLJt4FuUid?usp=drive_link 

3. You will need to create a folder called PyTorchSegmentation (Or another name you will remember). This folder should hold other folders titled ModelSaves, Segmentation, TAData, and Rellis3D.

   a. Place DeepLabV3 file in ModelSaves and replace the Rellis3D folder with the Rellis3D zip folder found in the drive link 

   b. After you create these folders, you will need to edit the file paths in these python files. Below is a list of each line that needs to edited in each file, and what to change it to.

      Main.py: edit lines 17, 18, 48, 227 and 265. Each of these have a portion that looks like "C:/Users/maste/Desktop/PyTorchSegmentation/Folder_name". Replace these with your path to the folder specified by "Folder_name"

      SegmentationTraining.py: edit lines 15, 53 and 59.
     


