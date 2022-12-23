import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import depth_reconstruction as depth

# Python script to use the Intel Real Sense camera

print("Environment Ready")

pipeline = rs.pipeline()
config = rs.config()
print("Pipeline is created")

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

#Low Resolution
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# High Resolution
# config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
# config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

# Start streaming
profile = pipeline.start(config)
# Set device settings
sensor = profile.get_device().query_sensors()[0]
sensor.set_option(rs.option.exposure, 40000)
sensor.set_option(rs.option.laser_power, 0)

templateHeight = 4
templateWidth = 4

#Specifications of IntelRealSense Camera used
fx = 596.230590820313
fy = 596.230590820313
B = 0.0549554191529751
cx = 319.500213623047
cy = 234.513290405273

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        
        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        infrared_frame1 = frames.get_infrared_frame(1)
        infrared_frame2 = frames.get_infrared_frame(2)

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        infrared_image1 = np.asanyarray(infrared_frame1.get_data())
        infrared_image2 = np.asanyarray(infrared_frame2.get_data())
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

        # depth_colormap_dim = depth_colormap.shape
        # color_colormap_dim = color_image.shape
        
        Estimator = depth.Estimator(infrared_image1, infrared_image2)
        Estimator.setCameraParameters(fx, fy, B, cx, cy)
        Estimator.calculateDisparity(templateWidth, templateHeight)
        
        cv.namedWindow('Disparity', cv.WINDOW_AUTOSIZE)
        cv.imshow('Disparity', Estimator.disparity)
        
        cv.waitKey(10)

finally:

    # Stop streaming
    pipeline.stop()