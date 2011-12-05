import ecto
from ecto_opencv.highgui import VideoCapture, imshow, FPSDrawer, MatPrinter, MatReader, imread
from ecto_opencv.imgproc import cvtColor, Conversion
from ecto_opencv.features2d import FAST, ORB, DrawKeypoints, Matcher, MatchRefinement, \
    MatchRefinementHSvd, MatchRefinement3d, MatchRefinementPnP, DrawMatches, KeypointsToMat
from ecto_opencv.calib import LatchMat, Select3d, Select3dRegion, PlaneFitter, PoseDrawer, DepthValidDraw, TransformCompose
from ecto_object_recognition.tod_detection import LSHMatcher
from ecto_registration import MatchStereoProj


class FeatureFinder(ecto.BlackBox):
    orb = ORB
    fast = FAST
    keypointsTo2d = KeypointsToMat
    select3d = Select3d
    image = ecto.Passthrough

    def declare_params(self, p):
        p.declare('use_fast', 'Use fast or not.', False)
        p.forward_all('orb')
        p.forward_all('fast')

    def declare_io(self, p, i, o):
        if p.use_fast:
            i.forward('mask', 'fast')
            i.forward('image', 'image', 'in')
        else:
            i.forward_all('orb')
        self.use_fast = p.use_fast
        i.forward('points3d', 'select3d')
        o.forward_all('orb')
        o.forward_all('keypointsTo2d')
        o.forward_all('select3d')

    def connections(self):
        graph = [self.orb['keypoints'] >> self.keypointsTo2d['keypoints'],
                self.keypointsTo2d['points'] >> self.select3d['points'], ]
        if self.use_fast:
            graph += [self.image[:] >> (self.fast['image'], self.orb['image']),
                      self.fast['keypoints'] >> self.orb['keypoints'],
                      ]
        return graph


class PlaneEstimator(ecto.BlackBox):
    #find a plane in the center region of the image.
    region = Select3dRegion
    plane_fitter = PlaneFitter
    flag = ecto.Passthrough

    def declare_params(self, p):
        p.forward_all('region')

    def declare_io(self, p, i, o):
        i.forward_all('region')
        i.forward('set', 'flag', cell_key='in')
        o.forward_all('plane_fitter')

    def connections(self):
        return [ self.region['points3d'] >> self.plane_fitter['points3d'],
                ]


class PoseEstimator3DProjImages(ecto.BlackBox):
    '''Estimates the pose of a reference frame to current frame, using 3D projection
    '''

    def declare_params(self, p):
        p.declare('show_matches', 'Display the matches.', False)
    def declare_io(self, p, i, o):
        self.gray_image = ecto.Passthrough('gray Input')
        self.gray_image_ref = ecto.Passthrough('gray Input from ref image')
        self.K = ecto.Passthrough('K')
        self.points3d = ecto.Passthrough('points3d')
        self.points3d_ref = ecto.Passthrough('points3d from ref image')
        self.depth_mask = ecto.Passthrough('mask')
        self.pose_estimation = MatchRefinementHSvd('Pose Estimation', reprojection_error=3, inlier_thresh=15)
        self.fps = FPSDrawer('FPS')
        self.tr = TransformCompose('Transform Composition')

        #inputs
        i.declare('K', self.K.inputs.at('in'))
        i.declare('image', self.gray_image.inputs.at('in'))
        i.declare('image_ref', self.gray_image_ref.inputs.at('in'))
        i.declare('mask', self.depth_mask.inputs.at('in'))
        i.declare('points3d', self.points3d.inputs.at('in'))
        i.declare('points3d_ref', self.points3d_ref.inputs.at('in'))
        i.declare('points3d', self.points3d.inputs.at('in'))
        i.declare('points3d_ref', self.points3d_ref.inputs.at('in'))

        #outputs
        o.declare('R', self.tr.outputs.at('R'))
        o.declare('T', self.tr.outputs.at('T'))
        o.declare('found', self.pose_estimation.outputs.at('found'))
        o.declare('debug_image', self.fps.outputs.at('image'))

    def configure(self, p, i, o):
        self.show_matches = p.show_matches

    def connections(self):
        n_features = 1000
        orb = ORB(n_features=n_features, n_levels=3, scale_factor=1.1)
        graph = [ self.gray_image[:] >> orb['image'],
                  self.points3d[:] >> orb['points3d'],
                ]

        orb_ref = ORB(n_features=n_features, n_levels=3, scale_factor=1.1)
        graph = [ self.gray_image_ref[:] >> orb_ref['image'],
                  self.points3d_ref[:] >> orb_ref['points3d'],
                ]

        matcher = LSHMatcher('LSH', n_tables=4, multi_probe_level=1, key_size=10, radius=70)
#        matcher = Matcher()
        graph += [ orb['descriptors'] >> matcher['test'],
                   orb_ref['descriptors'] >> matcher['train'],
                  ]

        #3d estimation
        pose_estimation = self.pose_estimation
        graph += [matcher['matches'] >> pose_estimation['matches'],
                  orb['points'] >> pose_estimation['test_2d'],
                  orb_ref['points'] >> pose_estimation['train_2d'],
                  orb['points3d'] >> pose_estimation['test_3d'],
                  orb_ref['points3d'] >> pose_estimation['train_3d'],
                  ]

        if self.show_matches:
            #display matches
            match_drawer = DrawMatches()
            graph += [pose_estimation['matches'] >> match_drawer['matches'],
                      pose_estimation['matches_mask'] >> match_drawer['matches_mask'],
                      orb['points'] >> match_drawer['test'],
                      orb_ref['points'] >> match_drawer['train'],
                      self.rgb_image[:] >> match_drawer['test_image'],
                      orb_ref['image'] >> match_drawer['train_image'],
                      match_drawer['output'] >> imshow(name='Matches')['']
                      ]

        tr = self.tr
        fps = self.fps
        # pose_draw = PoseDrawer()
        # graph += [train['R', 'T'] >> tr['R1', 'T1'],
        #           pose_estimation['R', 'T'] >> tr['R2', 'T2'],
        #           tr['R', 'T'] >> pose_draw['R', 'T'],
        #           pose_estimation['found'] >> pose_draw['trigger'],
        #           self.K[:] >> pose_draw['K'],
        #           self.rgb_image[:] >> pose_draw['image'],
        #           pose_draw['output'] >> fps[:],
        #           ]
        return graph


class PoseEstimator3DProj(ecto.BlackBox):
    '''Estimates the pose of a reference frame to current frame, using 3D projection
    '''
    def declare_params(self, p):
        p.declare('show_matches', 'Display the matches.', False)

    def declare_io(self, p, i, o):
        self.image = ecto.Passthrough('gray Input from test image')
        self.image_ref = ecto.Passthrough('gray Input from ref image')
        self.K = ecto.Passthrough('K')
        self.points = ecto.Passthrough('keypoints from test image')
        self.points_ref = ecto.Passthrough('keypoints from ref image')
        self.points3d = ecto.Passthrough('points3d from test image')
        self.points3d_ref = ecto.Passthrough('points3d from ref image')
        self.depth_mask = ecto.Passthrough('mask')
        self.desc = ecto.Passthrough('descriptors from test image')
        self.desc_ref = ecto.Passthrough('descriptors from ref image')
#        self.pose_estimation = MatchRefinementHSvd('Pose Estimation', reprojection_error=3, inlier_thresh=15)
#        self.pose_estimation = MatchRefinementHSvd('Pose Estimation')
        self.pose_estimation = MatchStereoProj('Pose Estimation', reprojection_error=3.0, n_iters = 1000)
        self.fps = FPSDrawer('FPS')
        self.tr = TransformCompose('Transform Composition')

        #inputs
        i.declare('K', self.K.inputs.at('in'))
        i.declare('image', self.image.inputs.at('in'))
        i.declare('image_ref', self.image_ref.inputs.at('in'))
        i.declare('mask', self.depth_mask.inputs.at('in'))
        i.declare('points', self.points.inputs.at('in'))
        i.declare('points_ref', self.points_ref.inputs.at('in'))
        i.declare('points3d', self.points3d.inputs.at('in'))
        i.declare('points3d_ref', self.points3d_ref.inputs.at('in'))
        i.declare('desc', self.desc.inputs.at('in'))
        i.declare('desc_ref', self.desc_ref.inputs.at('in'))

        #outputs
        o.declare('R', self.tr.outputs.at('R'))
        o.declare('T', self.tr.outputs.at('T'))
#        o.declare('found', self.pose_estimation.outputs.at('found'))
        o.declare('debug_image', self.fps.outputs.at('image'))

    def configure(self, p, i, o):
        self.show_matches = p.show_matches

    def connections(self):
        # LSH matcher, somewhat sensitive to parameters
#        matcher = LSHMatcher('LSH', n_tables=4, multi_probe_level=1, key_size=6, radius=55)  
         # this is a brute force matcher
        matcher = Matcher()  
        graph = [ self.desc[:] >> matcher['test'],
                   self.desc_ref[:] >> matcher['train'],
                  ]

        #3d estimation
        pose_estimation = self.pose_estimation
        graph += [matcher['matches'] >> pose_estimation['matches'],
                  self.points[:] >> pose_estimation['test_2d'],
                  self.points_ref[:] >> pose_estimation['train_2d'],
                  self.points3d[:] >> pose_estimation['test_3d'],
                  self.points3d_ref[:] >> pose_estimation['train_3d'],
                  self.K[:] >> pose_estimation['K']
                  ]

        if self.show_matches:
            #display matches
            match_drawer = DrawMatches()
            graph += [pose_estimation['matches'] >> match_drawer['matches'],
                      pose_estimation['matches_mask'] >> match_drawer['matches_mask'],
                      self.points[:] >> match_drawer['test'],
                      self.points_ref[:] >> match_drawer['train'],
                      self.image[:] >> match_drawer['test_image'],
                      self.image_ref[:] >> match_drawer['train_image'],
                      match_drawer['output'] >> imshow(name='Matches')['']
                      ]

        tr = self.tr
        fps = self.fps
        # pose_draw = PoseDrawer()
        # graph += [train['R', 'T'] >> tr['R1', 'T1'],
        #           pose_estimation['R', 'T'] >> tr['R2', 'T2'],
        #           tr['R', 'T'] >> pose_draw['R', 'T'],
        #           pose_estimation['found'] >> pose_draw['trigger'],
        #           self.K[:] >> pose_draw['K'],
        #           self.rgb_image[:] >> pose_draw['image'],
        #           pose_draw['output'] >> fps[:],
        #           ]
        return graph
