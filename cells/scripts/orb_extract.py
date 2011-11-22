#!/usr/bin/env python
import ecto
from ecto_opencv.highgui import VideoCapture, imshow, FPSDrawer
from ecto_opencv.features2d import ORB, DrawKeypoints
from ecto_opencv.imgproc import cvtColor, Conversion

from ecto_openni import OpenNICapture, DEPTH_RGB
from ecto_opencv.calib import DepthTo3d
from ecto.opts import scheduler_options, run_plasm

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

plasm.connect(img_src >> orb['image'],
              orb['keypoints'] >> draw_kpts['keypoints'],
              capture['image'] >> draw_kpts['image'],
              draw_kpts['image'] >> fps[:],
              fps[:] >> imshow('orb display', name='ORB')['image'],
              )


if __name__ == '__main__':
    from ecto.opts import doit
    doit(plasm, description='Computes the ORB feature and descriptor on an RGBD device.')

#sched = ecto.schedulers.Singlethreaded(plasm)
#sched.execute()

