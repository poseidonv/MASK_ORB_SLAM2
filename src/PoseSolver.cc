#include "PoseSolver.h"

namespace ORB_SLAM2
{
    void PoseSlover::SolvePose(vector<cv::KeyPoint> &vKPs1, vector<cv::KeyPoint> &vKPs2, vector<int> &vMatches12, cv::Mat &T21, cv::Mat &K)
    {
        vector<cv::Point2f> points1;
        vector<cv::Point2f> points2;
        for(size_t i = 0; i < vMatches12.size(); i++)
        {
            if(vMatches12[i] == -1)
                continue;
            
            points1.push_back(vKPs1[i].pt);
            points2.push_back(vKPs2[vMatches12[i]].pt);
        }

        cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);

        F.convertTo(F, CV_32FC1);
        cv::Mat E = K.t() * F * K;

        cv::Mat tmpK = K.clone();
        tmpK.convertTo(tmpK, CV_64FC1);
        E.convertTo(E, CV_64FC1);
        cv::Mat R, t;
        cv::recoverPose(E, points1, points2, tmpK, R, t);

        R.copyTo(T21.rowRange(0,3).colRange(0,3));
        t.copyTo(T21.rowRange(0,3).col(3));
        // cout<<"R: \n" <<R << endl <<"t: \n"<<t<<endl;
    }
}