//
// Created by lsf on 2023/5/11.
//


#include "Nanodet.h"

float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}


void activation_function_softmax(const float *src, float *dst, int length)
{
    const float alpha = *std::max_element(src, src + length);
    float denominator{0};
    for (int i = 0; i < length; ++i)
    {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
    }
}


static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v){
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f*s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
        case 0:r = v; g = t; b = p;break;
        case 1:r = q; g = v; b = p;break;
        case 2:r = p; g = v; b = t;break;
        case 3:r = p; g = q; b = v;break;
        case 4:r = t; g = p; b = v;break;
        case 5:r = v; g = p; b = q;break;
        default:r = 1; g = 1; b = 1;break;}
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}


static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}


NanoDet::NanoDet(const std::string &model_path, float score_threshold, float nms_threshold)
{
    // 1.创建OpenVINO Runtime Core对象
    ov::Core core;
    // 2.载入并编译模型
    ov::CompiledModel compile_model = core.compile_model(model_path, "CPU");
    // 3.创建推理请求
    infer_request_ = compile_model.create_infer_request();

    // 4. 初始化一些变量
    input_image_ = cv::Mat(input_height_, input_width_, CV_8UC3);
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;
    Boxes_.reserve(1000);
    center_priors_.reserve(2150);
    // 生成锚点
    generate_grid_center_priors();
}


void NanoDet::detect(cv::Mat &image, std::vector<Box>& boxes_res)
{
    // 1. 图像预处理
    preprocess(image);
    // 2. 推理
//    auto start = std::chrono::steady_clock::now();
    infer();
//    auto end = std::chrono::steady_clock::now();
//    std::chrono::duration<double> elapsed = end - start;
//    double time = 1000 * elapsed.count();
//    printf("Infer time = %.2f ms\n", time);
    // 3. 解码输出得到框
    decode_infer();
    // 4. 非极大抑制
    NMS(boxes_res);
    Boxes_.clear();
}


void NanoDet::draw(cv::Mat &image, std::vector<Box> &boxes_res)
{
    for(auto & ibox : boxes_res){
        float left = ibox.x1;
        float top = ibox.y1;
        float right = ibox.x2;
        float bottom = ibox.y2;
        int class_label = ibox.label;
        float score = ibox.score;
        cv::Scalar color;
        std::tie(color[0], color[1], color[2]) = random_color(class_label);
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), color, 2);

        auto name = class_labels_[class_label];
        auto caption = cv::format("%s %.2f", name.c_str(), score);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(left-3, top-33), cv::Point(left + text_width, top), color, -1);
        cv::putText(image, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
}


void NanoDet::preprocess(cv::Mat& image)
{
    // 通过双线性插值对图像进行resize
    float scale_x = (float)input_width_ / (float)image.cols;
    float scale_y = (float)input_height_ / (float)image.rows;
    float scale = std::min(scale_x, scale_y);

    // resize图像，源图像和目标图像几何中心的对齐
    i2d_[0] = scale;  i2d_[1] = 0;  i2d_[2] = (-scale * image.cols + input_width_ + scale  - 1) * 0.5;
    i2d_[3] = 0;  i2d_[4] = scale;  i2d_[5] = (-scale * image.rows + input_height_ + scale - 1) * 0.5;

    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d_);  // image to dst(network), 2x3 matrix
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i_);  // dst to image, 2x3 matrix
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);  // 计算一个反仿射变换

    // 对图像做平移缩放旋转变换,可逆
    cv::warpAffine(image, input_image_, m2x3_i2d, input_image_.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));
//    cv::imshow("debug", input_image_);
//    cv::waitKey(0);
}


void NanoDet::infer()
{
    // openvino 推理部分
    ov::element::Type input_type = ov::element::u8;
    ov::Shape input_shape = {1, static_cast<size_t>(input_height_), static_cast<size_t>(input_width_), 3};
    // 使用ov::Tensor包装图像数据，无需分配新内存
    ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, input_image_.data);
    infer_request_.set_input_tensor(input_tensor);
    infer_request_.infer();
    // 得到输出特征张量
    output_tensor_ = infer_request_.get_output_tensor();
    output_ptr_ = output_tensor_.data<float>();
}


void NanoDet::decode_infer()
{
    const int num_points = (int)center_priors_.size();
    const int num_channels = num_class_ + (reg_max_ + 1) * 4;

    for (int idx = 0; idx < num_points; idx++)
    {
        const int ct_x = center_priors_[idx].x;
        const int ct_y = center_priors_[idx].y;
        const int stride = center_priors_[idx].stride;

        float *ptr = output_ptr_ + idx * num_channels;
        int label = std::max_element(ptr, ptr + num_class_) - ptr;
        float score = ptr[label];

        if (score > score_threshold_)
        {
            float* bbox_pred = output_ptr_ + idx * num_channels + num_class_;
            Boxes_.emplace_back(disPred2Bbox(bbox_pred, label, score, ct_x, ct_y, stride));
        }
    }
}


Box NanoDet::disPred2Bbox(float* dfl_det, int label, float score, int x, int y, int stride) const
{
    float ct_x = x * stride;
    float ct_y = y * stride;
    std::vector<float> dis_pred;
    dis_pred.reserve(4);
    float dis_after_sm[reg_max_ + 1];
    for (int i = 0; i < 4; i++)
    {
        float dis = 0;
        activation_function_softmax(dfl_det + i * (reg_max_ + 1), dis_after_sm, reg_max_ + 1);
        for (int j = 0; j < reg_max_ + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        dis_pred[i] = dis;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float)input_width_);
    float ymax = (std::min)(ct_y + dis_pred[3], (float)input_height_);
    float image_x1 = d2i_[0] * xmin + d2i_[2];
    float image_y1 = d2i_[0] * ymin + d2i_[5];
    float image_x2 = d2i_[0] * xmax + d2i_[2];
    float image_y2 = d2i_[0] * ymax + d2i_[5];

    return Box{image_x1, image_y1, image_x2, image_y2, score, label};
}


void NanoDet::NMS(std::vector<Box>& boxes_res)
{
    std::sort(Boxes_.begin(), Boxes_.end(), [](Box& a, Box& b) {return a.score > b.score;});
    std::vector<bool> remove_flags(Boxes_.size());
    boxes_res.reserve(Boxes_.size());

    auto iou = [](const Box& a, const Box& b){
        float cross_left   = std::max(a.x1, b.x1);
        float cross_top    = std::max(a.y1, b.y1);
        float cross_right  = std::min(a.x2, b.x2);
        float cross_bottom = std::min(a.y2, b.y2);

        float cross_area = std::max(0.f, cross_right - cross_left) * std::max(0.f, cross_bottom - cross_top);
        float union_area = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1)
                           + std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1) - cross_area;
        if(cross_area == 0 || union_area == 0) return 0.f;
        return cross_area / union_area;
    };

    for(int i = 0; i < Boxes_.size(); ++i){
        if(remove_flags[i]) continue;

        auto& ibox = Boxes_[i];
        boxes_res.emplace_back(ibox);
        for(int j = i + 1; j < Boxes_.size(); ++j){
            if(remove_flags[j]) continue;

            auto& jbox = Boxes_[j];
            if(ibox.label == jbox.label){
                // class matched
                if(iou(ibox, jbox) >= nms_threshold_)
                    remove_flags[j] = true;
            }
        }
    }
//    printf(" Boxes_result.size = %d \n", (int)boxes_res.size());
}


void NanoDet::generate_grid_center_priors()
{
    for (int stride : strides_)
    {
        int feat_w = std::ceil((float)input_width_ / (float)stride);
        int feat_h = std::ceil((float)input_height_ / (float)stride);
        for (int y = 0; y < feat_h; y++)
        {
            for (int x = 0; x < feat_w; x++)
            {
                CenterPrior ct{};
                ct.x = x;
                ct.y = y;
                ct.stride = stride;
                center_priors_.push_back(ct);
            }
        }
    }
}

void NanoDet::benchmark(int loop_num) {
    int warm_up = 50;
    input_image_ = cv::Mat(320, 320, CV_8UC3, cv::Scalar(1, 1, 1));
    // warmup
    for (int i = 0; i < warm_up; i++)
    {
        infer();
    }
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < loop_num; i++)
    {
        infer();
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = 1000 * elapsed.count();
    printf("Average infer time = %.2f ms\n", time / loop_num);
}


