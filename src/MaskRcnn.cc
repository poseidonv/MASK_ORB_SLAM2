#include <c10/cuda/CUDAStream.h>
#include <torchvision/nms.h>

#include "MaskRcnn.h"

using namespace std;

namespace ORB_SLAM2
{
MaskRcnn::MaskRcnn(std::string model_path, int flag): mbNewImgFlag(false), mbNewSegImgFlag(false), mdevice(at::kCPU)
{
    bool cuda_availible = torch::cuda::is_available();
    std::cout<<"cuda availible: "<<cuda_availible<<std::endl;
    module = torch::jit::load(model_path);

    assert(module.buffers().size() > 0);
    cout<<(*begin(module.buffers())).device()<<endl;
    mdevice = (*begin(module.buffers())).device();
    cout<<mdevice<<endl;

    if(!flag)
    {
        mHeight = 376;
        mWidth = 1241;
    }
    else
    {
        mHeight = 480;
        mWidth = 640;
    }
}

// void MaskRcnn::SetTracker(Tracking* pTracker)
// {
//     mpTracker = pTracker;
// }

void MaskRcnn::Run()
{
    while (1)
    {   
        usleep(1);
        if(!IsNewImgArrived())
            continue;
        mvOutput.clear();
        bool flag = ProcessImage(mimage);
        at::Tensor tensor_image = ImageToTensor(mprocessed_image);
        c10::IValue output = Inference(tensor_image);
        ProcessOutput(output);
        // mNewImgFlag = false;
        // return mvOutput;
        SendSegImg();
        // break;
    }
}

bool MaskRcnn::SetImage(Frame* pFrame, cv::Mat& image)
{
    std::unique_lock<mutex> lock(mMutexGetNewImg);
    mpFrame = pFrame;
    cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    mimage = image.clone();
    mbNewImgFlag = true;
    return true;
}

bool MaskRcnn::IsNewImgArrived()
{
    std::unique_lock<mutex> lock(mMutexGetNewImg);
    if(mbNewImgFlag)
    {
        mbNewImgFlag = false;
        return true;
    }
    else
    {
        return false;
    }
}

void MaskRcnn::SendSegImg()
{
    std::unique_lock<mutex> lock(mMutexNewImgSegment);
    // mpTracker->mvSegments = mvOutput;
    mpFrame->mvSegments = mvOutput;
    mvOutput.clear();
    mbNewSegImgFlag = true;
}

bool MaskRcnn::ProcessImage(cv::Mat& image)
{
    // cv::resize(image, mprocessed_image, cv::Size(mWidth,mHeight));
    mprocessed_image = image.clone();
    cv::resize(mprocessed_image, mprocessed_image, cv::Size(mWidth,mHeight));

    if( mprocessed_image.channels() == 4)
    {
        cv::cvtColor(mprocessed_image, mprocessed_image, cv::COLOR_BGRA2BGR);
    }

    return true;
}

at::Tensor MaskRcnn::ImageToTensor(cv::Mat& processed_image)
{
    at::Tensor input = torch::from_blob(processed_image.data, {mHeight, mWidth, 3}, torch::kUInt8);
    // cout<<input.sizes()<<endl;
    input = input.to(mdevice, torch::kFloat).permute({2,0,1}).contiguous();
    
    return input;
}

c10::IValue MaskRcnn::Inference(at::Tensor& tensor_image)
{   
    // cout<<"input shape: "<<tensor_image.sizes()<<endl;
    c10::IValue output = module.forward({tensor_image});

    return output;
}

void MaskRcnn::ProcessOutput(c10::IValue& output)
{
    // mvOutput;

    auto outputs = output.toTuple()->elements();
    at::Tensor bbox = outputs[0].toTensor().to(at::kCPU), scores = outputs[1].toTensor().to(at::kCPU),
       labels = outputs[2].toTensor().to(at::kCPU), mask_probs = outputs[3].toTensor().to(at::kCPU);

    cv::Mat scores_mat = cv::Mat(scores.sizes()[0],1,CV_32FC1, cv::Scalar(0));
    std::memcpy((void *) scores_mat.data, scores.data_ptr<float>(), 4*scores.numel());
    // cv::sortIdx(score_mat, score_mat, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
    // cout<<"scores: "<<scores_mat<<endl;

    cv::Mat labelsmat = cv::Mat(labels.size(0), 1,CV_64FC1, cv::Scalar(0));
    std::memcpy((void *) labelsmat.data, labels.data_ptr<long>(), 8*labels.numel());

    // cout<<"labels: "<<endl;
    // for(int i = 0; i<labelsmat.rows; i++)
    // {
    //     cout<<int(labelsmat.at<long>(i,0))<<endl;
    // }

    cv::Mat boxes = cv::Mat(bbox.size(0), 4, CV_32FC1, cv::Scalar(0));
    // cout<<sizeof(bbox[0][0])<<endl;
    std::memcpy((void *) boxes.data, bbox.data_ptr<float>(), 4*bbox.numel());
    // cout<<"boxes: "<<boxes<<endl;

    at::Tensor img_masks = ProcessMaskProbs(bbox, mask_probs);
    cv::Mat mask = CombineMask(boxes, img_masks, labelsmat);
    cv::resize(mask, mask, cv::Size(mimage.cols, mimage.rows));
    // cv::imwrite("mask.jpg", mask);


    mvOutput.push_back(scores_mat);
    mvOutput.push_back(labelsmat);
    mvOutput.push_back(boxes);
    mvOutput.push_back(mask);
    // return vOutput;
    
}

at::Tensor MaskRcnn::ProcessMaskProbs(at::Tensor bbox, at::Tensor mask_probs)
{
    int n = bbox.sizes()[0];
    float scale_x, scale_y;
    scale_x = mprocessed_image.cols * 1.0 / mprocessed_image.cols;
    scale_y = mprocessed_image.rows * 1.0 / mprocessed_image.rows;
    at::Tensor scale = torch::tensor({scale_x, scale_y, scale_x, scale_y});
    vector<at::Tensor> vbbox = bbox.split(1,1);
    vbbox[0] = vbbox[0].mul(scale_x).clamp(0,mprocessed_image.cols);
    vbbox[1] = vbbox[1].mul(scale_y).clamp(0,mprocessed_image.rows);
    vbbox[2] = vbbox[2].mul(scale_x).clamp(0,mprocessed_image.cols);
    vbbox[3] = vbbox[3].mul(scale_y).clamp(0,mprocessed_image.rows);
    at::Tensor subw = (vbbox[2] - vbbox[0]).greater(0).expand({n,4});
    at::Tensor subh = (vbbox[3] - vbbox[1]).greater(0).expand({n,4});
    at::Tensor areas = (vbbox[2] - vbbox[0])*(vbbox[3] - vbbox[1]);
    // cout<<"areas: "<<endl<<areas<<endl;
    bbox = torch::cat(vbbox, 1);
    bbox = bbox.index(subw).reshape({-1,4}).index(subh).reshape({-1,4});
    // cout<<"bbox[0]: "<<bbox[0]<<endl;
    // cout<<"bbox[1]: "<<bbox[1]<<endl;

    int x0,y0,x1,y1;
    x0 = 0;
    y0 = 0;
    x1 = mprocessed_image.cols;
    y1 = mprocessed_image.rows;
    // cout<<"mask probs: "<<mask_probs[0][0]<<endl;
    at::Tensor img_x = torch::arange(x0, x1).to(at::kFloat) + 0.5;
    at::Tensor img_y = torch::arange(y0, y1).to(at::kFloat) + 0.5;
    img_x = (img_x - vbbox[0])/(vbbox[2] - vbbox[0])*2 - 1;
    img_y = (img_y - vbbox[1])/(vbbox[3] - vbbox[1])*2 - 1;
    at::Tensor gx = img_x.unsqueeze(1).expand({img_x.sizes()[0], img_y.sizes()[1], img_x.sizes()[1]});
    at::Tensor gy = img_y.unsqueeze(2).expand({img_x.sizes()[0], img_y.sizes()[1], img_x.sizes()[1]});
    at::Tensor grid = torch::stack({gx, gy}, 3);
    // cout<<"grid type: "<<grid.toString()<<endl;
    // cout<<"mask prob type: "<<mask_probs.toString()<<endl;
    at::Tensor img_masks = torch::grid_sampler(mask_probs, grid, 0, 0, false);
    img_masks = img_masks.squeeze(1);
    // cout<<img_masks.sizes()<<endl;
    // cout<<img_masks[0][400]<<endl;
    img_masks = img_masks.greater(0.5);

    return img_masks;
}

cv::Mat MaskRcnn::CombineMask(cv::Mat boxes,at::Tensor img_masks, cv::Mat labelsmat)
{
    cv::Mat mask = cv::Mat(mHeight, mWidth, CV_8UC3, cv::Scalar(0));
    cv::Mat mask0 = cv::Mat(mHeight, mWidth, CV_8UC1, cv::Scalar(0));

    vector<cv::Scalar> colors = {cv::Scalar(0,64,0),cv::Scalar(0,0,64), cv::Scalar(0,64,64), cv::Scalar(64,0,64),
                                cv::Scalar(0,128,64),cv::Scalar(128,0,64), cv::Scalar(64,128,0), cv::Scalar(64,128,128),
                                cv::Scalar(256,0,64),cv::Scalar(128,256,64),cv::Scalar(256,64,128)};
    map<int, string> classname;
    classname[0] = "person";
    classname[56] = "chair";
    classname[62] = "tv";
    classname[66] = "keyboard";
    classname[64] = "mouse";
    classname[39] = "bottle";
    // int img_area = mprocessed_image.cols*mprocessed_image.rows;
    for(int i=0; i<boxes.rows; i++)
    {
        // cout<<boxes.row(i)<<endl;
        int label = i+1;
        // if(int(labelsmat.at<long>(i,0)) == 0)
        // {
        //     std::memcpy((void *) mask0.data, img_masks[i].data_ptr<bool>(), 1*img_masks[i].numel());
        //     mask.setTo(colors[1], mask0);
        //     int x0,y0,x1,y1;
        //     x0 = int(boxes.at<float>(i,0));
        //     y0 = int(boxes.at<float>(i,1));
        //     x1 = int(boxes.at<float>(i,2));
        //     y1 = int(boxes.at<float>(i,3));
        //     int area = (x1 - x0)*(y1 - y0);
        //     float font_size = area * 2.27 /img_area;
        //     if(font_size < 0.1) font_size = 0.1;

        //     cv::rectangle(mask, cv::Rect(cv::Point2i(x0, y0), cv::Point2i(x1, y1)) ,cv::Scalar(255,255,255));
        //     // int classnum = int(labelsmat.at<long>(i,0));
        //     // cout<<classnum<<endl;
        //     // cv::putText(mask, classname[classnum], cv::Point2i(x0, int(y0+font_size*10)), cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(128,128,128));
        // }
        std::memcpy((void *) mask0.data, img_masks[i].data_ptr<bool>(), 1*img_masks[i].numel());
        mask.setTo(cv::Scalar(label,label,label), mask0);
    }

    return mask.clone();
}

}