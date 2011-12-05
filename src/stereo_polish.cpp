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


#include <registration/pe.h>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <iostream>
#include <opencv2/core/eigen.hpp>

#include "g2o/core/block_solver.h"
#include "g2o/core/graph_optimizer_sparse.h"
#include "g2o/core/solver.h"
#include "g2o/core/structure_only_solver.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/icp/types_icp.h"
#include "g2o/sba.h"


using namespace Eigen;
using namespace std;
using namespace g2o;

namespace pe
{
  // constructor
  StereoPolish::StereoPolish(int n_iters, bool motion_only)
  {
    numIters = n_iters;
  }

  void StereoPolish::polish(const matches_t &matches, 
			    const cv::Mat &train_kpts, const cv::Mat &query_kpts,
			    const cv::Mat &train_pts, const cv::Mat &query_pts,
			    const cv::Mat &K, const double baseline,
			    Eigen::Matrix3d &R, Eigen::Vector3d &t)
  {
    // make sure we're using floats
    if (train_kpts.depth() != CV_32F ||
	query_kpts.depth() != CV_32F ||
	train_pts.depth() != CV_32F)
      throw std::runtime_error("Expected input to be of floating point (CV_32F)");

    int nmatch = matches.size();
    cout << "[polish] matches: " << nmatch << endl; 

    Matrix3f Kf;
    cv2eigen(K,Kf);		    // camera matrix
    float fb = Kf(0,0)*baseline; // focal length times baseline

    // set up optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setMethod(g2o::SparseOptimizer::LevenbergMarquardt);
    optimizer.setVerbose(true);
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    // we should use a dense solver if we're going for Schur complements
    if (0)
    {
      linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    }
    else
    {
      linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
    }

    g2o::BlockSolver_6_3 *solver_ptr = 
      new g2o::BlockSolver_6_3(&optimizer, linearSolver);
    optimizer.setSolver(solver_ptr);

    // set up camera params
    g2o::VertexSCam::setKcam(Kf(0,0), Kf(1,1), Kf(0,2), Kf(1,2), baseline);

    // set up the camera vertices, just two
    Quaterniond r;
    r.setIdentity();
    Vector3d tr;
    tr.setZero();
    g2o::SE3Quat pose(r,tr);	// Zero pose
    g2o::VertexSCam * v_se3 = new g2o::VertexSCam();
    v_se3->setId(0);
    v_se3->estimate() = pose;
    v_se3->setAll(); // set aux transforms
    v_se3->setFixed(true);
    optimizer.addVertex(v_se3);

    g2o::SE3Quat pose2(R,t);	// Offset pose
    g2o::VertexSCam * v2_se3 = new g2o::VertexSCam();
    v2_se3->setId(1);
    v2_se3->estimate() = pose2;
    v2_se3->setAll(); // set aux transforms
    optimizer.addVertex(v2_se3);

    // set up edges to points
    int point_id = 2;
    for (int i=0; i<nmatch; i++)
      {
	const float* ti = train_pts.ptr<float>(matches[i].trainIdx);
	const float* qi = query_pts.ptr<float>(matches[i].queryIdx);
	const float* ti2 = train_kpts.ptr<float>(matches[i].trainIdx);
	const float* qi2 = query_kpts.ptr<float>(matches[i].queryIdx);

	Vector3d pt(ti[0],ti[1],ti[2]);
	g2o::VertexPointXYZ * v_p = new g2o::VertexPointXYZ();
	v_p->setId(point_id);
	v_p->setMarginalized(true);
	v_p->estimate() = pt;
	optimizer.addVertex(v_p);

	// add new edges
        g2o::Edge_XYZ_VSC * e = new g2o::Edge_XYZ_VSC();
        e->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p);
        e->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_se3);
        Vector3d z(ti2[0],ti2[1],fb/ti[2]);
        e->measurement() = z;
        e->inverseMeasurement() = -z;
        e->information() = Matrix3d::Identity();
        // TODO
        //e->setRobustKernel(ROBUST_KERNEL);
        e->setHuberWidth(1);
        optimizer.addEdge(e);

        g2o::Edge_XYZ_VSC * e2 = new g2o::Edge_XYZ_VSC();
        e2->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p);
        e2->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v2_se3);
        Vector3d z2(qi2[0],qi2[1],fb/qi[2]);
        e2->measurement() = z2;
        e2->inverseMeasurement() = -z2;
        e2->information() = Matrix3d::Identity();
        // TODO
        //e->setRobustKernel(ROBUST_KERNEL);
        e2->setHuberWidth(1);
        optimizer.addEdge(e2);
      }	

    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    cout << "[polish] Performing full BA:" << endl;
    optimizer.optimize(numIters);
    
    // return results
    pose2 = v2_se3->estimate();
    AngleAxisd aa(pose2.rotation());
    cout << "[stereo_polish] " << endl
	 << pose2.translation().transpose() << endl
	 << aa.angle()*180.0/M_PI << "   " << aa.axis().transpose() << endl;

  }


  void
  sba_process_impl(const Eigen::SparseMatrix<int> &x, const Eigen::SparseMatrix<int> & y,
                   const Eigen::SparseMatrix<int> &disparity, const Eigen::Matrix3d & K,
                   const VectorQuaterniond & quaternions, const VectorVector3d & Ts,
                   const VectorVector3d & in_point_estimates, VectorVector3d & out_point_estimates)
  {
    g2o::SparseOptimizer optimizer;
    optimizer.setMethod(g2o::SparseOptimizer::LevenbergMarquardt);
    optimizer.setVerbose(false);
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    if (0)
    {
      linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    }
    else
    {
      linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
    }

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(&optimizer, linearSolver);

    optimizer.setSolver(solver_ptr);

    double baseline = 0.075; // 7.5 cm baseline

    // set up camera params
    g2o::VertexSCam::setKcam(K(0, 0), K(1, 1), K(0, 2), K(1, 2), baseline);

    // set up the camera vertices
    unsigned int vertex_id = 0;
    for (size_t i = 0; i < quaternions.size(); ++i)
    {
      g2o::SE3Quat pose(quaternions[i], Ts[i]);

      g2o::VertexSCam * v_se3 = new g2o::VertexSCam();

      v_se3->setId(vertex_id);
      v_se3->estimate() = pose;
      v_se3->setAll(); // set aux transforms

      optimizer.addVertex(v_se3);
      vertex_id++;
    }

    int point_id = vertex_id;

    tr1::unordered_map<int, int> pointid_2_trueid;

    // add point projections to this vertex
    for (size_t i = 0; i < in_point_estimates.size(); ++i)
    {
      g2o::VertexPointXYZ * v_p = new g2o::VertexPointXYZ();

      v_p->setId(point_id);
      v_p->setMarginalized(true);
      v_p->estimate() = in_point_estimates.at(i);

      // First, make sure, the point is visible in several views
      {
        unsigned int count = 0;
        for (size_t j = 0; j < quaternions.size(); ++j)
        {
          if ((x.coeff(j, i) == 0) && (y.coeff(j, i) == 0) && (disparity.coeff(j, i) == 0))
            continue;
          ++count;
          if (count >= 2)
            break;
        }
        if (count < 2)
          continue;
      }
      // Add the different views to the optimizer
      for (size_t j = 0; j < quaternions.size(); ++j)
      {
        // check whether the point is visible in that view
        if ((x.coeff(j, i) == 0) && (y.coeff(j, i) == 0) && (disparity.coeff(j, i) == 0))
          continue;

        g2o::Edge_XYZ_VSC * e = new g2o::Edge_XYZ_VSC();

        e->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p);

        e->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(j)->second);

        Vector3d z(x.coeff(j, i), y.coeff(j, i), disparity.coeff(j, i));
        e->measurement() = z;
        e->inverseMeasurement() = -z;
        e->information() = Matrix3d::Identity();

        // TODO
        //e->setRobustKernel(ROBUST_KERNEL);
        e->setHuberWidth(1);

        optimizer.addEdge(e);
      }

      optimizer.addVertex(v_p);

      pointid_2_trueid.insert(make_pair(point_id, i));

      ++point_id;
    }

    optimizer.initializeOptimization();

    optimizer.setVerbose(true);

    g2o::StructureOnlySolver<3> structure_only_ba;

    cout << "Performing full BA:" << endl;
    optimizer.optimize(5);

    // Set the computed points in the final data structure
    out_point_estimates = in_point_estimates;
    for (tr1::unordered_map<int, int>::iterator it = pointid_2_trueid.begin(); it != pointid_2_trueid.end(); ++it)
    {
      g2o::HyperGraph::VertexIDMap::iterator v_it = optimizer.vertices().find(it->first);

      if (v_it == optimizer.vertices().end())
      {
        cerr << "Vertex " << it->first << " not in graph!" << endl;
        exit(-1);
      }

      g2o::VertexPointXYZ * v_p = dynamic_cast<g2o::VertexPointXYZ *>(v_it->second);

      if (v_p == 0)
      {
        cerr << "Vertex " << it->first << "is not a PointXYZ!" << endl;
        exit(-1);
      }
      out_point_estimates[it->second] = v_p->estimate();
    }

    // Set the unchange
  }

} // ends namespace pe
