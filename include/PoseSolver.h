#ifndef POSESOLVER_H
#define POSESOLVER_H

#include <vector>
#include <map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv/cvaux.h>
#include <opencv/cxcore.hpp>

using namespace std;

namespace ORB_SLAM2
{
    class PoseSlover
    {
    public:
        static void SolvePose(vector<cv::KeyPoint> &vKPs1, vector<cv::KeyPoint> &vKPs2, vector<int> &vMatches12, cv::Mat &T21, cv::Mat &K);
    };
}

#endif