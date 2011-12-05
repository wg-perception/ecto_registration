#include <ecto/ecto.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <registration/pe.h>

#include <numeric>
#include <algorithm>

using ecto::tendrils;

typedef std::vector<cv::KeyPoint> kpts_t;
typedef std::vector<cv::Point2f> points_t;
typedef std::vector<cv::Point3f> points3d_t;
typedef std::vector<cv::DMatch> matches_t;
typedef cv::Mat_<uchar> mask_t;
namespace
{
  struct select_train
  {
    int
    operator()(const cv::DMatch& m) const
    {
      return m.trainIdx;
    }
  };
  struct select_test
  {
    int
    operator()(const cv::DMatch& m) const
    {
      return m.queryIdx;
    }
  };

  template<typename PointT>
  struct select_train_
  {
    select_train_(const cv::Mat& points)
        :
          points(points)
    {
    }

    PointT
    operator()(const cv::DMatch& m) const
    {
      return points.at<PointT>(m.trainIdx);
    }
    const cv::Mat& points;
  };
  template<typename PointT>
  struct select_test_
  {
    select_test_(const cv::Mat& points)
        :
          points(points)
    {
    }

    PointT
    operator()(const cv::DMatch& m) const
    {
      PointT x = points.at<PointT>(m.queryIdx);
      return x;
    }
    const cv::Mat& points;
  };

  template<typename PointT>
  struct subtract_
  {
    subtract_(const PointT& p)
        :
          p(p)
    {
    }

    PointT
    operator()(const PointT& x) const
    {
      return x - p;
    }
    const PointT& p;
  };
  typedef select_test_<cv::Point3f> select_test_3d;
  typedef select_test_<cv::Point2f> select_test_2d;
  typedef select_train_<cv::Point3f> select_train_3d;
  typedef select_train_<cv::Point2f> select_train_2d;

  struct nan_predicate
  {
    nan_predicate(const cv::Mat& train, const cv::Mat& test)
        :
          train(train),
          test(test)
    {
    }

    inline bool
    is_nan_safe(const cv::Point3f& p) const
    {
      return p.x != p.x || p.y != p.y || p.z != p.z;
    }

    bool
    operator()(const cv::DMatch& m) const
    {
      return is_nan_safe(train.at<cv::Point3f>(m.trainIdx)) || is_nan_safe(test.at<cv::Point3f>(m.queryIdx));
    }
    const cv::Mat& train;
    const cv::Mat& test;
  };

  template<int dist>
  struct match_distance_predicate
  {

    bool
    operator()(const cv::DMatch& m) const
    {
      return m.distance > dist;
    }
  };

  template<typename T>
  struct mask_predicate_
  {
    mask_predicate_(T mask)
    {
      it = mask;
    }
    bool
    operator()(const cv::DMatch&)
    {
      return 0 == *(it++);

    }
    T it;
  };
  template<typename T>
  mask_predicate_<T>
  mask_predicate(const T& it)
  {
    return mask_predicate_<T>(it);
  }

  cv::Point3f
  demean_points(const std::vector<cv::Point3f>& points, std::vector<cv::Point3f>& out)
  {
    cv::Point3f p;
    p = std::accumulate(points.begin(), points.end(), p);
    p *= 1.f / points.size();
    cv::Mat tm(points);
    out.resize(points.size());
    std::transform(points.begin(), points.end(), out.begin(), subtract_<cv::Point3f>(p));
    return p;
  }

  template<typename T>
  int
  sign(T f)
  {
    if (f > 0)
      return 1;
    else
      return -1;
  }

}


struct MatchStereoProj
{
  typedef MatchStereoProj C;

  static void
  declare_params(tendrils& p)
  {
    p.declare(&C::n_iters, "n_iters", "number of ransac iterations", 1000);
    p.declare(&C::reprojection_error, "reprojection_error", "error threshold in pixels", 5);
    p.declare(&C::min_inliers, "min_inliers", "minimum number of inliers", 30);
    p.declare(&C::inlier_thresh, "inlier_thresh", "The inlier threshold of pose found.", 20);
  }
  static void
  declare_io(const tendrils& p, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare(&C::K, "K", "Camera matrix.").required(true);
    inputs.declare(&C::baseline, "baseline", "Stereo baseline.");
    inputs.declare(&C::train_2d, "train_2d", "The 2d training points.").required(true);
    inputs.declare(&C::test_2d, "test_2d", "The 2d test points.").required(true);
    inputs.declare(&C::train_3d, "train_3d", "The 3d training points.").required(true);
    inputs.declare(&C::test_3d, "test_3d", "The 3d test points.").required(true);
    inputs.declare(&C::matches_in, "matches", "The descriptor matches.").required(true);
    outputs.declare(&C::matches_out, "matches", "The verified matches.");
    outputs.declare(&C::matches_mask, "matches_mask", "The matches mask, same size as the original matches.");
    outputs.declare(&C::R_out, "R");
    outputs.declare(&C::T_out, "T");
    outputs.declare(&C::found_out, "found");
  }

  int
  process(const tendrils&inputs, const tendrils& outputs)
  {
    *found_out = false;
    //    std::cout << "Input matches: " << matches_in->size() << std::endl;

    pe::PoseEstimator posest = pe::PoseEstimator(*n_iters, false, 3.0, 0.1, *reprojection_error, *reprojection_error);

    int nmatch;
    nmatch = posest.estimate(*matches_in, *train_2d, *test_2d, *train_3d, 
			     *test_3d, *K, 0.075);

    //need to preallocate the mask!
    cv::Mat inl(1, posest.inliers.size(), CV_8U, 1);
    *matches_out = posest.inliers;
    //    *matches_out = *matches_in;
    *matches_mask = inl;
    std::cout << "Inliers: " << nmatch << std::endl;
    return ecto::OK;


#if 0
    *R_out = R;
    *T_out = T;
    *matches_out = good_matches;
    *matches_mask = inlier_mask;
    float inlier_percentage = 100 * float(demeaned_train_pts.size()) / good_matches.size();
    *found_out = inlier_percentage > *inlier_thresh && *min_inliers/2 < demeaned_test_pts.size();
#endif

    std::cout << "Found matches: " << matches_out->size() << std::endl;

    return ecto::OK;
  }
  ecto::spore<cv::Mat> K, train_2d, test_2d, test_3d, train_3d, R_out, T_out;
  ecto::spore<matches_t> matches_in, matches_out;
  ecto::spore<cv::Mat> matches_mask;
  ecto::spore<bool> found_out;
  ecto::spore<unsigned> n_iters, min_inliers;
  ecto::spore<float> reprojection_error,inlier_thresh, baseline;

};
ECTO_CELL(ecto_registration, MatchStereoProj, "MatchStereoProj",
          "A feature descriptor pose estimator, stereo projection");
