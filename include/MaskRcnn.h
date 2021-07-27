#ifndef MASK_RCNN_H
#define MASK_RCNN_H

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <assert.h>
#include <memory>
#include <ctime>
#include <vector>
#include <map>
#include <mutex>

#include <torch/torch.h>
// #include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/script.h>
#include <torchvision/vision.h>
// #include <torchvision/nms.h>

// #include "Tracking.h"
#include "Frame.h"

using namespace std;

namespace ORB_SLAM2
{
// class Tracking;
class Frame;

class MaskRcnn
{
public:
    MaskRcnn(std::string model_path, int flag = 0);

    void Run();

    bool SetImage(Frame* pFrame, cv::Mat& image);

    bool ProcessImage(cv::Mat& image);

    at::Tensor ImageToTensor(cv::Mat& processed_image);

    c10::IValue Inference(at::Tensor& tensor_image);

    void ProcessOutput(c10::IValue& output);

    at::Tensor ProcessMaskProbs(at::Tensor bbox, at::Tensor mask_probs);

    cv::Mat CombineMask(cv::Mat boxes, at::Tensor img_masks, cv::Mat labelsmat);

    // void SetTracker(Tracking* pTracker);

    bool IsNewImgArrived();

    void SendSegImg();

    std::mutex mMutexGetNewImg;
    std::mutex mMutexNewImgSegment;

    bool mbNewImgFlag;
    bool mbNewSegImgFlag;
private:
    // Tracking* mpTracker;
    Frame* mpFrame;
    torch::jit::script::Module module;
    int mHeight;
    int mWidth;
    c10::Device mdevice;
    cv::Mat mimage;
    cv::Mat mprocessed_image;
    vector<cv::Mat> mvOutput;
};

}

#endif