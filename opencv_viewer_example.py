## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#Setup Aruco Detector
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
arucoParams = cv2.aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
         
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #ArUco Detection
        corners, ids, rejected = arucoDetector.detectMarkers(color_image)
        color_image = cv2.aruco.drawDetectedMarkers(color_image,corners,ids)


        # ================================
        # DRAWING
        # ================================

       #depthIntrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        """  
        for cornerSet, i in corners:
            print(f"ID: {ids[i]}\n")
            for corner in cornerSet[0, ...]:
                (x, y) = corner
                z = depth_frame.get_distance(x, y)
                p = rs.rs2_deproject_pixel_to_point(depthIntrinsics, [x, y], z) 
                
                
                     depthIntrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
   
    # 遍历所有检测到的标记
        for marker_index, marker_corners in enumerate(corners):
        # 遍历当前 ArUco 标记的每个角点
         for corner in marker_corners[0]:
            # 获取像素坐标 (x, y)
            x, y = int(corner[0]), int(corner[1])
            print(x)

            # 获取像素 (x, y) 处的深度值
            depth = depth_frame.get_distance(x, y)
            

            # 将像素坐标反投影到3D空间
            point_3d = rs.rs2_deproject_pixel_to_point(depthIntrinsics, [x, y], depth)

            # 输出每个角点的 3D 坐标
            print(f"标记 {ids[marker_index]} 在像素 ({x}, {y}) 处的角点的3D坐标: {point_3d}") 


            
    # 遍历所有检测到的标记
        for marker_index, marker_corners in enumerate(corners):
        # 遍历当前 ArUco 标记的每个角点
         for corner in marker_corners[0]:
            # 获取像素坐标 (x, y)
            x, y = int(corner[0]), int(corner[1])
            # 获取像素 (x, y) 处的深度值
            depth = depth_frame.get_distance(x, y)
            if depth > 0:  # Ensure we have valid depth
                        # Deproject pixel to 3D point
                        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
                        print(f"Marker {ids[marker_index]} at pixel ({x}, {y}) has 3D coordinates: {point_3d}")
            else:
                        print(f"No valid depth at pixel ({x}, {y}) for marker {ids[marker_index]}")
                """

 # ================================
    
    # Align depth frame to color frame
        align = rs.align(rs.stream.color)
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        for marker_index, maker_corners in enumerate (corners):
             
             corners_np = np.array(maker_corners[0])

             center_x = np.mean(corners_np[: , 0])
             center_y = np.mean(corners_np[: , 1])


            # Cast to integer for depth lookup
            #int_center_x = int(center_x)
             #int_center_y = int(center_y)
            
             depth = depth = depth_frame.get_distance(center_x, center_y)
             point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [center_x, center_y], depth)

             print(f"{ids[marker_index]}: {point_3d}")
   
       

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
