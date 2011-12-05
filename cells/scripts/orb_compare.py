#!/usr/bin/env python
import ecto
from ecto_opencv.highgui import VideoCapture, imshow, FPSDrawer
from ecto_opencv.features2d import ORB, DrawKeypoints
from ecto_opencv.imgproc import cvtColor, Conversion

from ecto_openni import OpenNICapture, DEPTH_RGB
from ecto_opencv.calib import DepthTo3d
from ecto.opts import scheduler_options, run_plasm

from image_pipeline import LatchMat

n_feats = 500
plasm = ecto.Plasm()

#setup the input source, grayscale conversion
capture = OpenNICapture(stream_mode=DEPTH_RGB, registration=True, sync=True)
rgb2gray = cvtColor (flag=Conversion.RGB2GRAY)

plasm.connect(capture['image'] >> rgb2gray ['image'])

#convenience variable for the grayscale
img_src = rgb2gray['image']

# calculate 3d points
#depthTo3d = DepthTo3d()
#plasm.connect(capture['depth'] >> depthTo3d['depth']
#)

# connect up the test ORB
orb = ORB(n_features=n_feats)
draw_kpts = DrawKeypoints()
fps = FPSDrawer()
disp = imshow('orb display', name='ORB', triggers=dict(save=ord('s')))

plasm.connect(img_src >> orb['image'],
              orb['keypoints'] >> draw_kpts['keypoints'],
              capture['image'] >> draw_kpts['image'],
              draw_kpts['image'] >> fps[:],
              fps[:] >> disp['image'],
              )

# connect up the reference ORB
orb_ref = ORB(n_features=n_feats)
draw_kpts_ref = DrawKeypoints()
latch_ref = LatchMat(init=True)

plasm.connect(img_src >> latch_ref['input'],
              latch_ref['output'] >> orb_ref['image'],
              disp['save'] >> latch_ref['set'],
              orb_ref['keypoints'] >> draw_kpts_ref['keypoints'],
              latch_ref['output'] >> draw_kpts_ref['image'],
              draw_kpts_ref['image'] >> imshow('reference orb display', name='ORB reference')['image'],
              )


if __name__ == '__main__':
    from ecto.opts import doit
    doit(plasm, description='Computes the ORB feature and descriptor on an RGBD device.')

#sched = ecto.schedulers.Singlethreaded(plasm)
#sched.execute()

