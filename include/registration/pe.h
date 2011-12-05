/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/


#ifndef _PE_H_
#define _PE_H_

#include <boost/shared_ptr.hpp>
#include <cv.h>
#include <cstdlib>
#include <math.h>
#include <Eigen/Dense>
#include <image_pipeline/pinhole_camera_model.h>


typedef std::vector<cv::KeyPoint> kpts_t;
typedef std::vector<float> depth_t;
typedef std::vector<cv::Point2f> points_t;
typedef std::vector<cv::Point3f> points3d_t;
typedef std::vector<cv::DMatch> matches_t;

namespace pe
{
  /// A class that estimates camera pose from features in image frames.
  class PoseEstimator
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // needed for 16B alignment

    enum MethodType
    {
      SFM = 0,
      PnP = 1,
      Stereo = 2
    };

    MethodType usedMethod;

    // true if we want to LM polish the result
    bool polish;

    /// number of RANSAC iterations
    int numRansac;
    
    /// Whether to do windowed or whole-image matching.
    int windowed;

    /// \brief Initialize a pose estimator
    /// NRansac is number of ransac iterations
    /// LMpolish is true for bundle-adjusting the result
    /// maxrange is max Z dist on points (in meters) to participate in pose hypotheses
    /// minpdist is the minimum distance of pixels for hypotheses, in percent of image width
    /// maxidx and maxidd are inlier bounds for feature point and disparity projections
    PoseEstimator(int NRansac, bool LMpolish, double maxrange, double minpdist,
                  double maxidx, double maxidd);
    ~PoseEstimator() { }

    /// Maximum range for RANSAC point matches.
    double maxMatchRange;

    /// Minimum pixel distance for hypotheses
    double minPDist;

    /// Maximum dist for inliers in pixels.
    double maxInlierXDist, maxInlierDDist;

    /// \brief Uses RANSAC to find best inlier count from provided matches, 
    /// optionally polishes the result.
    /// Depth is interpreted as disparities via focal length and baseline
    int estimate(const matches_t &matches, 
		 const cv::Mat &train_kpts, const cv::Mat &query_kpts,
		 const cv::Mat &train_pts, const cv::Mat &query_pts,
                 const cv::Mat &K, const double baseline = 0);

    /// Get the method used for inlier detection.
    MethodType getMethod() const {return usedMethod;};

    // transform
    Eigen::Matrix3f rot; ///< Rotation matrix of camera between the frames.
    Eigen::Vector3f trans; ///< Translation of the camera between the frames.

    // inliers
    matches_t inliers;
  };

  /// Polishing a stereo frame-frame match
  class StereoPolish
  {
  public:
    StereoPolish(int n_iters, bool motion_only); // number of LM iterations
    void polish(const matches_t &matches, 
		const cv::Mat &train_kpts, const cv::Mat &query_kpts,
		const cv::Mat &train_pts, const cv::Mat &query_pts,
		const cv::Mat &K, const double baseline ,
		Eigen::Matrix3d &R, Eigen::Vector3d &t);
    int numIters;
    bool motionOnly;
  };
    

} // ends namespace pe
#endif // _PE_H_
