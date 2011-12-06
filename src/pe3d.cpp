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

using namespace Eigen;
using namespace std;

namespace pe
{

  // constructor
  PoseEstimator::PoseEstimator(int NRansac, bool LMpolish, double maxrange,
                               double minpdist, double maxidx, double maxidd)
  {
    numRansac = NRansac;
    maxMatchRange = maxrange;
    maxInlierXDist = maxidx;
    maxInlierDDist = maxidd;
    minPDist = minpdist;	// minimum distance between pixels for hypothesis, in percent of full image width
  }


  //
  // find the best estimate for a geometrically-consistent match
  // assumes stereo points have been calculated, and matches filtered,
  //   and frames set up
  // uses the SVD procedure for aligning point clouds
  //   SEE: Arun, Huang, Blostein: Least-Squares Fitting of Two 3D Point Sets
  // final SBA polishing step
  //

  // assumes cv::Mats are of type <float>
  int PoseEstimator::estimate(const matches_t &matches, 
                              const cv::Mat &train_kpts, const cv::Mat &query_kpts,
                              const cv::Mat &train_pts, const cv::Mat &query_pts,
			      const cv::Mat &K, const double baseline)
  {
    // make sure we're using floats
    if (train_kpts.depth() != CV_32F ||
	query_kpts.depth() != CV_32F ||
	train_pts.depth() != CV_32F ||
	query_pts.depth() != CV_32F)
      throw std::runtime_error("Expected input to be of floating point (CV_32F)");

    int nmatch = matches.size();

    cout << endl << "[pe3d] Matches: " << nmatch << endl;

    // set up data structures for fast processing
    // indices to good matches

    std::vector<Eigen::Vector3f> t3d, q3d;
    std::vector<Eigen::Vector2f> t2d, q2d;
    std::vector<int> m;		// keep track of matches 

    for (int i=0; i<nmatch; i++)
      {
	const float* ti = train_pts.ptr<float>(matches[i].trainIdx);
	const float* qi = query_pts.ptr<float>(matches[i].queryIdx);
	const float* ti2 = train_kpts.ptr<float>(matches[i].trainIdx);
	const float* qi2 = query_kpts.ptr<float>(matches[i].queryIdx);

	// make sure we have points within range; eliminates NaNs as well
        if (matches[i].distance < 60 && // TODO: get this out of the estimator
	    ti[2] < maxMatchRange && 
	    qi[2] < maxMatchRange)
	  {
	    m.push_back(i);
	    t2d.push_back(Eigen::Vector2f(ti2[0],ti2[1]));
	    q2d.push_back(Eigen::Vector2f(qi2[0],qi2[1]));
	    t3d.push_back(Eigen::Vector3f(ti[0],ti[1],ti[2]));
	    q3d.push_back(Eigen::Vector3f(qi[0],qi[1],qi[2]));
          }
      }


    nmatch = m.size();
    cout << "[pe3d] Filtered matches: " << nmatch << endl;

    if (nmatch < 3) return 0;   // can't do it...

    int bestinl = 0;
    Matrix3f Kf;
    cv::cv2eigen(K,Kf);		    // camera matrix
    float fb = Kf(0,0)*baseline; // focal length times baseline
    float maxInlierXDist2 = maxInlierXDist*maxInlierXDist;
    float maxInlierDDist2 = maxInlierDDist*maxInlierDDist;

    // set up minimum hyp pixel distance
    float minpdist = minPDist * Kf(0,2) * 2.0;

    // RANSAC loop
    int numchoices = 1 + numRansac / 10;
    for (int ii=0; ii<numRansac; ii++) 
      {
        // find a candidate
	int k;
        int a=rand()%nmatch;
	int b;
	for (k=0; k<numchoices; k++)
	  {
	    b=rand()%nmatch;
	    if (a!=b && (t2d[a]-t2d[b]).norm() > minpdist
		     && (q2d[a]-q2d[b]).norm() > minpdist)
	      // TODO: add distance check
	      break;
	  }
	if (k >= numchoices) continue;
        int c;
	for (k=0; k<numchoices+numchoices; k++)
	  {
	    c=rand()%nmatch;
	    if (c!=b && c!=a && (t2d[a]-t2d[c]).norm() > minpdist
		&& (t2d[b]-t2d[c]).norm() > minpdist
		&& (q2d[a]-q2d[c]).norm() > minpdist
		&& (q2d[b]-q2d[c]).norm() > minpdist)
	      // TODO: add distance check
	      break;
	  }
	if (k >= numchoices+numchoices) continue;

        // get centroids
        Vector3f p0a = t3d[a];
        Vector3f p0b = t3d[b];
        Vector3f p0c = t3d[c];

        Vector3f p1a = q3d[a];
        Vector3f p1b = q3d[b];
        Vector3f p1c = q3d[c];

        Vector3f c0 = (p0a+p0b+p0c)*(1.0/3.0);
        Vector3f c1 = (p1a+p1b+p1c)*(1.0/3.0);

        // subtract out
        p0a -= c0;
        p0b -= c0;
        p0c -= c0;
        p1a -= c1;
        p1b -= c1;
        p1c -= c1;

        Matrix3f Hf = p1a*p0a.transpose() + p1b*p0b.transpose() +
	              p1c*p0c.transpose();
	Matrix3d H = Hf.cast<double>();

#if 0
        cout << p0a.transpose() << endl;
        cout << p0b.transpose() << endl;
        cout << p0c.transpose() << endl;
#endif

        // do the SVD thang
        JacobiSVD<Matrix3d> svd(H, ComputeFullU | ComputeFullV);
        Matrix3d V = svd.matrixV();
        Matrix3d R = V * svd.matrixU().transpose();          
        double det = R.determinant();
        //ntot++;
        if (det < 0.0)
          {
            //nneg++;
            V.col(2) = V.col(2)*-1.0;
            R = V * svd.matrixU().transpose();
          }

	Vector3d cd0 = c0.cast<double>();
	Vector3d cd1 = c1.cast<double>();
        Vector3d tr = cd0-R*cd1;    // translation 

	//      cout << "[PE test] R: " << endl << R << endl;
	//	cout << "[PE test] t: " << tr.transpose() << endl;

        Vector3f trf = tr.cast<float>();      // convert to floats
        Matrix3f Rf = R.cast<float>();
	Rf = Kf*Rf;
	trf = Kf*trf;

        // find inliers, based on image reprojection
        int inl = 0;
        for (int i=0; i<nmatch; i++)
          {
            const Vector3f &pt1 = q3d[i];
            Vector3f ipt = Rf*pt1+trf; // transform query point
            float iz = 1.0/ipt.z();
	    Vector2f &kp = t2d[i];
	    //	    cout << kp.transpose() << " " << pt1.transpose() << " " << ipt.transpose() << endl;
            float dx = kp.x() - ipt.x()*iz;
            float dy = kp.y() - ipt.y()*iz;
            float dd = fb/t3d[i].z() - fb/ipt.z(); // diff of disparities, could pre-compute t3d
            if (dx*dx < maxInlierXDist2 && dy*dy < maxInlierXDist2
                && dd*dd < maxInlierDDist2)
               //              inl+=(int)fsqrt(ipt.z()); // clever way to weight closer points
              inl++;
          }
        
        if (inl > bestinl) 
          {
            bestinl = inl;
            trans = tr.cast<float>();      // convert to floats
            rot = R.cast<float>();
          }

      }

    cout << "[pe3d] Best inliers: " << bestinl << endl;
//    printf("Total ransac: %d  Neg det: %d\n", ntot, nneg);

    // reduce matches to inliers
    matches_t inls;    // temporary for current inliers
    inliers.clear();
    Matrix3f Rf = Kf*rot;
    Vector3f trf = Kf*trans;

    //    cout << "[pe3e] R: " << endl << rot << endl;
    cout << "[pe3d] t: " << trans.transpose() << endl;

    AngleAxisf aa;
    aa.fromRotationMatrix(rot);
    cout << "[pe3d] AA: " << aa.angle()*180.0/M_PI << "   " << aa.axis().transpose() << endl;

    for (int i=0; i<nmatch; i++)
      {
	Vector3f &pt1 = q3d[i];
        Vector3f ipt = Rf*pt1+trf; // transform query point
        float iz = 1.0/ipt.z();
	Vector2f &kp = t2d[i];
	//	cout << kp.transpose() << " " << pt1.transpose() << " " << ipt.transpose() << endl;
        float dx = kp.x() - ipt.x()*iz;
        float dy = kp.y() - ipt.y()*iz;
        float dd = fb/t3d[i].z() - fb/ipt.z(); // diff of disparities, could pre-compute t3d

        if (dx*dx < maxInlierXDist2 && dy*dy < maxInlierXDist2 && 
            dd*dd < maxInlierDDist2)
	  inls.push_back(matches[m[i]]); 
      }

    cout << "[pe3d] Final inliers: " << inls.size() << endl;

    // polish with SVD
    if (polish)
      {
	Matrix3d Rd = rot.cast<double>();
	Vector3d trd = trans.cast<double>();
	StereoPolish pol(5,false);
	pol.polish(inls,train_kpts,query_kpts,train_pts,query_pts,K,baseline,
	   Rd,trd);

	AngleAxisd aa;
	aa.fromRotationMatrix(Rd);
	cout << "[pe3d] Polished t: " << trd.transpose() << endl;
	cout << "[pe3d] Polished AA: " << aa.angle()*180.0/M_PI << "   " << aa.axis().transpose() << endl;

	int num = inls.size();
	// get centroids
	Vector3f c0(0,0,0);
	Vector3f c1(0,0,0);
	for (int i=0; i<num; i++)
	  {
	    c0 += t3d[i];
	    c1 += q3d[i];
	  }

	c0 = c0 / (float)num;
	c1 = c1 / (float)num;

        // subtract out and create H matrix
	Matrix3f Hf;
	Hf.setZero();

	for (int i=0; i<num; i++)
	  {
	    Vector3f p0 = t3d[i]-c0;
	    Vector3f p1 = q3d[i]-c1;
	    Hf += p1*p0.transpose();
	  }

	Matrix3d H = Hf.cast<double>();

        // do the SVD thang
        JacobiSVD<Matrix3d> svd(H, ComputeFullU | ComputeFullV);
        Matrix3d V = svd.matrixV();
        Matrix3d R = V * svd.matrixU().transpose();          
        double det = R.determinant();
        //ntot++;
        if (det < 0.0)
          {
            //nneg++;
            V.col(2) = V.col(2)*-1.0;
            R = V * svd.matrixU().transpose();
          }

	Vector3d cd0 = c0.cast<double>();
	Vector3d cd1 = c1.cast<double>();
        Vector3d tr = cd0-R*cd1;    // translation 


	aa.fromRotationMatrix(R);
	cout << "[pe3d] t: " << tr.transpose() << endl;
	cout << "[pe3d] AA: " << aa.angle()*180.0/M_PI << "   " << aa.axis().transpose() << endl;

#if 0
        // system
        SysSBA sba;
        sba.verbose = 0;

        // set up nodes
        // should have a frame => node function        
        Vector4d v0 = Vector4d(0,0,0,1);
        Quaterniond q0 = Quaternion<double>(Vector4d(0,0,0,1));
        sba.addNode(v0, q0, f0.cam, true);
        
        Quaterniond qr1(rot);   // from rotation matrix
        Vector4d temptrans = Vector4d(trans(0), trans(1), trans(2), 1.0);

        //        sba.addNode(temptrans, qr1.normalized(), f1.cam, false);
        qr1.normalize();
        sba.addNode(temptrans, qr1, f1.cam, false);

        int in = 3;
        if (in > (int)inls.size())
          in = inls.size();

        // set up projections
        for (int i=0; i<(int)inls.size(); i++)
          {
            // add point
            int i0 = inls[i].queryIdx;
            int i1 = inls[i].trainIdx;
            Vector4d pt = query_pts[i0];
            sba.addPoint(pt);
            
            // projected point, ul,vl,ur
            Vector3d ipt;
            ipt(0) = f0.kpts[i0].pt.x;
            ipt(1) = f0.kpts[i0].pt.y;
            ipt(2) = ipt(0)-f0.disps[i0];
            sba.addStereoProj(0, i, ipt);

            // projected point, ul,vl,ur
            ipt(0) = f1.kpts[i1].pt.x;
            ipt(1) = f1.kpts[i1].pt.y;
            ipt(2) = ipt(0)-f1.disps[i1];
            sba.addStereoProj(1, i, ipt);
          }

        sba.huber = 2.0;
        sba.doSBA(5,10e-4,SBA_DENSE_CHOLESKY);
        int nbad = sba.removeBad(2.0);
//        cout << endl << "Removed " << nbad << " projections > 2 pixels error" << endl;
        sba.doSBA(5,10e-5,SBA_DENSE_CHOLESKY);

//        cout << endl << sba.nodes[1].trans.transpose().head(3) << endl;

        // get the updated transform
	      trans = sba.nodes[1].trans.head(3);
	      Quaterniond q1;
	      q1 = sba.nodes[1].qrot;
	      rot = q1.toRotationMatrix();

        // set up inliers
        inliers.clear();
        for (int i=0; i<(int)inls.size(); i++)
          {
            ProjMap &prjs = sba.tracks[i].projections;
            if (prjs[0].isValid && prjs[1].isValid) // valid track
              inliers.push_back(inls[i]);
          }
#if 0
        printf("Inliers: %d   After polish: %d\n", (int)inls.size(), (int)inliers.size());
#endif

#endif

      }

    inliers = inls;
    return (int)inls.size();
  }
} // ends namespace pe
