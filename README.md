This is the repository for the UWF Sevil Research Group's Traversability Assessment of Unstructured Environments Through Image Segmentation and Path Planning. 

Our goal is to segment an image of an outdoor off road location into traversable and non-traversable surfaces. Then, the system will plan a viable path through the area and the platform will execute the path.
The steps to run these files and view the results yourself are listed below. These are subject to change as research progresses.

Version 1.0: 
We used the PyCharm IDE to work on this project, and the steps will reference this IDE.

1. Download Files from GitHub. Also download the Rellis3D dataset and deeplab model we used from this drive link:
   https://drive.google.com/drive/folders/12t654a10328USRkx45VYmUrLJt4FuUid?usp=drive_link 

2. You will need to create a folder called PyTorchSegmentation (Or another name you will remember). This folder should hold other folders titled ModelSaves, Segmentations, TAData, and Rellis3D. Avoid placing this in an existing system folder such as Users, as this can cause issues with permissions and access later in the code. We reccommend following this filepath: "C:/PyTorchSegmentation/" where "C:" is replaced by your drive of choice. 

   a. Place DeepLabV3 file in ModelSaves and replace the Rellis3D folder with the Rellis3D zip folder found in the drive link

   b. Create 2 folders within Segmentations titled tempExamples4, and AStarPathingImages (If you change this, you will get errors asking you to correct file paths in lines 139 and 168).

   c. After you create these folders, you will need to edit the file paths in these python files. Below is a list of each line that needs to edited in each file, and what to change it to.

   Main.py: edit lines 17, 18, 48, 227 and 265. Each of these have a portion that looks like "C:/Python/PyTorchSegmentation/Folder_name". Replace these with your path to the folder specified by "Folder_name", if you chose a different location than the reccomended one.

3. After these files are created, you will need to ensure that each of the libraries included at the top of the file are installed. If you are using PyCharm and they arent installed, you can click the error on the uninstalled reference and select "Install library".

Once each of these steps is completed, your Main.py will be ready to run.

If you experience any issues or find an error, please email cxh1@students.uwf.edu with the subject line "Traversability Assessment Questions". 
     


