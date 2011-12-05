#!/usr/bin/env python
import ecto
from ecto_opencv.highgui import VideoCapture, imshow, FPSDrawer
from ecto_opencv.features2d import ORB, DrawKeypoints, KeypointsToMat
from ecto_opencv.imgproc import cvtColor, Conversion

from ecto_openni import OpenNICapture, DEPTH_RGB
from ecto_opencv.calib import DepthTo3d, DepthMask
from ecto.opts import scheduler_options, run_plasm

from image_pipeline import LatchMat
import ecto_registration
from registration import PoseEstimator3DProj, FeatureFinder, RotateZ

n_feats = 1000
plasm = ecto.Plasm()

#setup the input source, grayscale conversion
capture = OpenNICapture(stream_mode=DEPTH_RGB, registration=True, sync=True)
rgb2gray = cvtColor (flag=Conversion.RGB2GRAY)

plasm.connect(capture['image'] >> rgb2gray ['image'])

#convenience variable for the grayscale
img_src = rgb2gray['image']

# calculate 3d points
depthTo3d = DepthTo3d()
plasm.connect(capture['depth'] >> depthTo3d['depth'],
              capture['K'] >> depthTo3d['K']
              )

# connect up the test ORB
orb = FeatureFinder('ORB test', n_features=n_feats, n_levels=6, scale_factor=1.2, thresh=100, use_fast=False)

plasm.connect(img_src >> orb['image'],
              depthTo3d['points3d'] >> orb['points3d'],
              )

# connect up the reference ORB
rot_image = RotateZ()
rot_depth = RotateZ()
rot_depthTo3d = DepthTo3d()
orb_ref = FeatureFinder('ORB ref', n_features=n_feats, n_levels=6, scale_factor=1.2, thresh=100, use_fast=False)

plasm.connect(img_src >> rot_image['image'],
              capture['depth'] >> rot_depth['image'],
              rot_depth['rotated_image'] >> rot_depthTo3d['depth'],
              capture['K'] >> rot_depthTo3d['K'],
              rot_depthTo3d['points3d'] >> orb_ref['points3d'],
              rot_image['rotated_image'] >> orb_ref['image'],
              )


# put in the estimator
posest = PoseEstimator3DProj(show_matches=True)
kmat = KeypointsToMat()
kmat_ref = KeypointsToMat()

plasm.connect(img_src >> posest['image'],
              rot_depth['rotated_image'] >> posest['image_ref'],
              orb['keypoints'] >> kmat['keypoints'],
              kmat['points'] >> posest['points'],
              orb_ref['keypoints'] >> kmat_ref['keypoints'],
              kmat_ref['points'] >> posest['points_ref'],
              orb['points3d'] >> posest['points3d'],
              orb_ref['points3d'] >> posest['points3d_ref'],
              orb['descriptors'] >> posest['desc'],
              orb_ref['descriptors'] >> posest['desc_ref'],
              capture['K'] >> posest['K']
              )

if __name__ == '__main__':
    from ecto.opts import doit
    doit(plasm, description='Pose estimation on an RGBD device.')

#sched = ecto.schedulers.Singlethreaded(plasm)
#sched.execute()

