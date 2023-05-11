//
// Created by lsf on 2023/5/11.
//

#include "Nanodet.h"


int image_demo(const std::shared_ptr<NanoDet>& detector, const char* imagepath)
{
    cv::Mat image = cv::imread(imagepath);
    if (image.empty())
    {
        fprintf(stderr, "cv::imread %s failed.\n", imagepath);
        return -1;
    }
    std::vector<Box> boxes;
    detector->detect(image, boxes);
    detector->draw(image, boxes);
    cv::imshow("Nanodet", image);
    cv::waitKey(0);
    boxes.clear();
    return 0;
}

int webcam_demo(const std::shared_ptr<NanoDet>& detector, int cam_id)
{
    cv::Mat image;
    cv::VideoCapture cap(cam_id);
    std::vector<Box> boxes;
    while (true)
    {
        bool ret = cap.read(image);
        if (!ret) {
            fprintf(stderr, "VideoCapture %d read failed.\n", cam_id);
            return 0;
        }
        auto start = std::chrono::steady_clock::now();
        detector->detect(image, boxes);
        detector->draw(image, boxes);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double time = 1000 * elapsed.count();
        printf("Detect time = %.2f ms\n", time);
        cv::imshow("Nanodet", image);
        cv::waitKey(1);
        boxes.clear();
    }
}

int video_demo(const std::shared_ptr<NanoDet>& detector, const char* path)
{
    cv::Mat image;
    cv::VideoCapture cap(path);
    std::vector<Box> boxes;
    while (true)
    {
        bool ret = cap.read(image);
        if (!ret) {
            fprintf(stderr, "Video %s read failed.\n", path);
            return 0;
        }
        auto start = std::chrono::steady_clock::now();
        detector->detect(image, boxes);
        detector->draw(image, boxes);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double time = 1000 * elapsed.count();
        printf("Detect time = %.2f ms\n", time);
        cv::imshow("Nanodet", image);
        cv::waitKey(1);
        boxes.clear();
    }
}

void benchmark(const std::shared_ptr<NanoDet>& detector)
{
    int loop_num = 1000;
    detector->benchmark(loop_num);
}


int main(int argc, char** argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id; \n For image demo, mode=1, path=xxx/xxx/*.jpg; \n For video, mode=2; \n For benchmark, mode=3 path=0.\n", argv[0]);
        return -1;
    }

    std::cout << "Start init model." << std::endl;
    auto detector = std::make_shared<NanoDet>("nanodet_int8.xml", 0.4, 0.5);
    std::cout << "Init model success." << std::endl;

    int mode = atoi(argv[1]);
    switch (mode)
    {
    case 0:{
        const char* images = argv[2];
        image_demo(detector, images);
        break;
        }
    case 1:{
        int cam_id = atoi(argv[2]);
        webcam_demo(detector, cam_id);
        break;
        }
    case 2:{
        const char* path = argv[2];
        video_demo(detector, path);
        break;
        }
    case 3:{
        benchmark(detector);
        break;
        }
    default:{
        fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id; \n For image demo, mode=1, path=xxx/xxx/*.jpg; \n For video, mode=2; \n For benchmark, mode=3 path=0.\n", argv[0]);
        break;
        }
    }
}
