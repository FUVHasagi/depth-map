import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
import glob


# Load in color and depth image to create the point cloud
imagesColor = glob.glob('images/stereoLeft/o3d_input/*.png')
imagesDepth = glob.glob('images/disp_map/*.png')
for imgColor, imgDepth in zip(imagesColor, imagesDepth):
    color_raw = o3d.io.read_image(imgColor)
    depth_raw = o3d.io.read_image(imgDepth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)
    print(rgbd_image)

    # # Camera intrinsic parameters built into Open3D for Prime Sense
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    # Create the point cloud from images and camera intrisic parameters
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
        
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])
    

    o3d.io.write_point_cloud("images/o3d_output/" + (imgColor.split('\\')[-1]).split('.')[0] + "_ptcloud.ply", pcd)
    new_pcd = o3d.io.read_point_cloud("images/o3d_output/" + (imgColor.split('\\')[-1]).split('.')[0] + "_ptcloud.ply")
    o3d.visualization.draw_geometries([new_pcd])
    # Plot the images
    plt.subplot(1, 2, 1)
    plt.title('Grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()