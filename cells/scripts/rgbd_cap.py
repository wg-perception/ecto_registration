#!/usr/bin/env python

#
# capture depth and rgb images from an RGBD-type device
# works with ASUS, PrimeSense, and Kinect devices
# registers depth to image (in driver)
#

import ecto
from ecto_opencv.highgui import imshow
from ecto_opencv.calib import DepthTo3d
from image_pipeline import Rectifier, StereoModelLoader, DepthRegister, CameraModelToCv, CV_INTER_NN
from ecto_openni import Capture, OpenNICapture, DEPTH_RGB, DEPTH_IR, RGB, IR, IRGamma, enumerate_devices
from ecto_object_recognition.conversion import MatToPointCloudXYZRGB
from ecto_pcl import PointCloudT2PointCloud, CloudViewer, XYZRGB

print enumerate_devices()

openni_reg = True
openni_sync = True
capture = OpenNICapture(stream_mode=DEPTH_RGB, registration=openni_reg, synchronization=openni_sync, latched=False)
#capture = Capture(registration=openni_reg, synchronize=openni_sync)

plasm = ecto.Plasm()

camera_converter = CameraModelToCv()

plasm.connect(
#    stereo_model['left_model'] >> camera_converter['camera'],
    capture['image'] >> imshow(name='Original')['image'],
#    capture['depth'] >> imshow(name='Original Depth')['image']
    )
    
if __name__ == '__main__':
    from ecto.opts import doit
    doit(plasm, description='Capture RGBD depth and RGB.',locals=vars())
