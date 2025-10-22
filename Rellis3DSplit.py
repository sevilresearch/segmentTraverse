from shutil import copyfile

tempRellis = "./TempRellis/"

for split in "train", "val", "test":
    #Images
    imageSplitFile = open(tempRellis + "Rellis_3D_image_split/" + split + ".lst", "r")

    for line in imageSplitFile:
        imagePath, annotationPath = line.split(" ")

        copyfile(tempRellis + "Rellis_3D_pylon_camera_node/Rellis-3D/" + imagePath, "./Datasets/Rellis3D/Images/Images/" + split + "/" + imagePath.split("/")[0] + "-" + imagePath.split("/")[2].split("-")[0].split("e")[1] + ".jpg")
        copyfile(tempRellis + "Rellis_3D_pylon_camera_node_label_id/Rellis-3D/" + annotationPath.rstrip(), "./Datasets/Rellis3D/Images/Annotations/" + split + "/" + imagePath.split("/")[0] + "-" + imagePath.split("/")[2].split("-")[0].split("e")[1] + ".png")

        #print(imagePath, annotationPath)
        print()

        print(tempRellis + "Rellis_3D_pylon_camera_node/Rellis-3D/" + imagePath, "./Datasets/Rellis3D/Images/Images/" + split + "/" + imagePath.split("/")[0] + "-" + imagePath.split("/")[2].split("-")[0].split("e")[1] + ".jpg")
        print(tempRellis + "Rellis_3D_pylon_camera_node_label_id/Rellis-3D/" + annotationPath.rstrip(), "./Datasets/Rellis3D/Images/Annotations/" + split + "/" + imagePath.split("/")[0] + "-" + imagePath.split("/")[2].split("-")[0].split("e")[1] + ".png")

        print(tempRellis + "Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/" + imagePath.split("/")[0] + "/os1_cloud_node_kitti_bin/" + imagePath.split("/")[2].split("-")[0].split("e")[1] + ".bin", "./Datasets/Rellis3D/Lidar/Lidar/" + split + "/" + imagePath.split("/")[0] + "-" + imagePath.split("/")[2].split("-")[0].split("e")[1] + ".bin")
        print(tempRellis + "Rellis_3D_os1_cloud_node_semantickitti_label_id/Rellis-3D/" + imagePath.split("/")[0] + "/os1_cloud_node_semantickitti_label_id/" + imagePath.split("/")[2].split("-")[0].split("e")[1] + ".label", "./Datasets/Rellis3D/Lidar/Annotations/" + split + "/" + imagePath.split("/")[0] + "-" + imagePath.split("/")[2].split("-")[0].split("e")[1] + ".label")

        copyfile(tempRellis + "Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/" + imagePath.split("/")[0] + "/os1_cloud_node_kitti_bin/" + imagePath.split("/")[2].split("-")[0].split("e")[1] + ".bin", "./Datasets/Rellis3D/Lidar/Lidar/" + split + "/" + imagePath.split("/")[0] + "-" + imagePath.split("/")[2].split("-")[0].split("e")[1] + ".bin")
        copyfile(tempRellis + "Rellis_3D_os1_cloud_node_semantickitti_label_id/Rellis-3D/" + imagePath.split("/")[0] + "/os1_cloud_node_semantickitti_label_id/" + imagePath.split("/")[2].split("-")[0].split("e")[1] + ".label", "./Datasets/Rellis3D/Lidar/Annotations/" + split + "/" + imagePath.split("/")[0] + "-" + imagePath.split("/")[2].split("-")[0].split("e")[1] + ".label")

        #print(imagePath, annotationPath)