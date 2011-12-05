//
// rotates an RGB/D image by 180 degrees, for testing

#include <ecto/ecto.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <numeric>
#include <algorithm>

using ecto::tendrils;


struct Rotate_Z
{
  typedef Rotate_Z C;

  static void
  declare_params(tendrils& p)
  {
  }

  static void
  declare_io(const tendrils& p, tendrils& inputs, tendrils& outputs)
  {
    inputs.declare(&C::image_in, "image", "Camera image");
    outputs.declare(&C::image_out, "rotated_image", "Rotated image");
  }

  int
  process(const tendrils&inputs, const tendrils&outputs)
  {
    cv::Mat image = *image_in;
    cv::Mat rotated;
    cv::flip(image,rotated,-1);	// flip around both axes
    *image_out = rotated;
    return ecto::OK;
  }
  ecto::spore<cv::Mat> image_in, depth_in, image_out, depth_out;
  ecto::spore<float> angle;

};
ECTO_CELL(ecto_registration, Rotate_Z, "RotateZ",
          "Rotate images around the Z axis");
