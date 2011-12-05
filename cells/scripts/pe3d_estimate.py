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
from registration import PoseEstimator3DProj, FeatureFinder

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
depthMask = DepthMask()
plasm.connect(capture['depth'] >> depthTo3d['depth'],
              capture['depth'] >> depthMask['depth'],
              capture['K'] >> depthTo3d['K']
              )

# connect up the test ORB
orb = FeatureFinder('ORB test', n_features=n_feats, n_levels=6, scale_factor=1.2, thresh=100, use_fast=False)
draw_kpts = DrawKeypoints()
fps = FPSDrawer()
disp = imshow('orb display', name='ORB', triggers=dict(save=ord('s')))

plasm.connect(img_src >> orb['image'],
              depthTo3d['points3d'] >> orb['points3d'],
#              depthMask['mask'] >> orb['mask'],
              orb['keypoints'] >> draw_kpts['keypoints'],
              capture['image'] >> draw_kpts['image'],
              draw_kpts['image'] >> fps[:],
              fps[:] >> disp['image'],
              )

# connect up the reference ORB
orb_ref = FeatureFinder('ORB test', n_features=n_feats, n_levels=6, scale_factor=1.2, thresh=100, use_fast=False)
draw_kpts_ref = DrawKeypoints()
latch_ref = LatchMat(init=True)
latch_ref_3d = LatchMat(init=True)
latch_ref_mask = LatchMat(init=True)

plasm.connect(img_src >> latch_ref['input'],
              disp['save'] >> latch_ref['set'],
              depthTo3d['points3d'] >> latch_ref_3d['input'],
              disp['save'] >> latch_ref_3d['set'],
              depthMask['mask'] >> latch_ref_mask['input'],
              disp['save'] >> latch_ref_mask['set'],
              latch_ref['output'] >> orb_ref['image'],
              latch_ref_3d['output'] >> orb_ref['points3d'],
#              latch_ref_mask['output'] >> orb_ref['mask'],
              latch_ref_mask['output'] >> imshow('ref mask', name='Reference image mask')['image'],
              depthMask['mask'] >> imshow('mask', name='image mask')['image'],
              orb_ref['keypoints'] >> draw_kpts_ref['keypoints'],
              latch_ref['output'] >> draw_kpts_ref['image'],
              draw_kpts_ref['image'] >> imshow('reference orb display', name='ORB reference')['image'],
              )


# put in the estimator
posest = PoseEstimator3DProj(show_matches=True)
kmat = KeypointsToMat()
kmat_ref = KeypointsToMat()

plasm.connect(img_src >> posest['image'],
              latch_ref['output'] >> posest['image_ref'],
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

