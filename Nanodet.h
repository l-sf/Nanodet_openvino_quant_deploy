//
// Created by lsf on 2023/5/11.
//

#ifndef NANODET_OPENVINO_H
#define NANODET_OPENVINO_H


#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <memory>
#include <functional>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <inference_engine.hpp>


struct Box
{
    float x1, y1, x2, y2, score;
    int label;
    
    Box() = default;
    Box(float x1, float y1, float x2, float y2, float score, int label):
        x1(x1), y1(y1), x2(x2), y2(y2), score(score), label(label) {}
};

struct CenterPrior
{
    int x;
    int y;
    int stride;
};

class NanoDet
{
public:
    explicit NanoDet(const std::string &model_path, float score_threshold = 0.4, float nms_threshold = 0.5);

    ~NanoDet() = default;

    void detect(cv::Mat& image, std::vector<Box>& boxes_res);

    void draw(cv::Mat& image, std::vector<Box>& boxes_res);

    void benchmark(int loop_num = 1000);

private:
    std::string input_name_ = "image";
    std::string output_name_ = "output";
    int input_width_ = 320;
    int input_height_ = 320;
    float score_threshold_ = 0.4;
    float nms_threshold_ = 0.5;
    int num_class_ = 80; // number of classes. 80 for COCO
    int reg_max_ = 7; // `reg_max` set in the training config. Default: 7.
    std::vector<int> strides_ = {8, 16, 32, 64}; // strides of the multi-level feature.

    void preprocess(cv::Mat& image);

    void infer();

    void decode_infer();

    void NMS(std::vector<Box>& boxes_res);

    void generate_grid_center_priors();

    ov::InferRequest infer_request_;
    ov::Tensor output_tensor_;
    cv::Mat input_image_;
    float i2d_[6]{}, d2i_[6]{};
    float* output_ptr_{};
    std::vector<Box> Boxes_;
    std::vector<CenterPrior> center_priors_;

    Box disPred2Bbox(float* dfl_det, int label, float score, int x, int y, int stride) const;

    std::vector<std::string> class_labels_ {
       "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
       "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
       "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
       "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
       "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
       "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
       "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
       "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
       "hair drier", "toothbrush"};

};


#endif //NANODET_OPENVINO_H
