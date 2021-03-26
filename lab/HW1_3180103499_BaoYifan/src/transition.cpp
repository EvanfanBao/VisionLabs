#include "transition.hpp"
#define TRANS_NUM  6  //其实是5个

using namespace cv;
using namespace std;

//调用5种传递 转场效果 用于开头显示图片
void transition(cv::VideoWriter& writer, const cv::Mat& img)
{
    static transition_effect* transition_array[TRANS_NUM] = {  transition_effect5,
                                                                    transition_effect2,
                                                                    transition_effect3,
                                                                    transition_effect4,
                                                                    transition_effect1
                                                                    };
    static int i;
    (*(transition_array[i++]))(writer, img);
    if (TRANS_NUM == i)
    {
        i = 0;
    }
}

void transition_effect1(cv::VideoWriter& writer, const cv::Mat& img)
{
    Mat img_translation(img.size(), img.type());
    for (int i = 1; i <= FPS; ++i)
    {
        img_translation.setTo(cv::Scalar(0, 0, 0));
        cv::Rect rect(0, img.rows * (1 - (float)i / (float)FPS), img.cols, img.rows * (float)i / (float)FPS);
        img.rowRange(0, img.rows * (float)i / (float)FPS).copyTo(img_translation(rect));
        writer<<img_translation;
    }
}

void transition_effect2(cv::VideoWriter& writer, const cv::Mat& img)
{
    const int interval = FPS * 2;
    cv::Mat img_masked(img.size(), img.type());
    cv::Mat mask(img.size(), img.type());
    for (int i = 0; i < FPS; ++i)
    {
        cv::Mat img_tmp;
        int whiteWidth = i * 2;
        mask.setTo(cv::Scalar(1.0, 1.0, 1.0));

        for (int j = 0; j < mask.rows / interval; ++j)
        {
            img_tmp = mask.rowRange(j * interval + whiteWidth, (j + 1) * interval);
            img_tmp.setTo(cv::Scalar(0.0,0.0,0.0));
        }
        if ((int)(mask.rows / interval) * interval + whiteWidth < mask.rows)
        {
            img_tmp = mask.rowRange((int)(mask.rows / interval) * interval + whiteWidth, mask.rows);
            img_tmp.setTo(cv::Scalar(0.0,0.0,0.0));
        }

        cv::multiply(img, mask, img_masked);
        writer<<img_masked;
    }
}
//高斯模糊
void transition_effect3(cv::VideoWriter& writer, const cv::Mat& img)
{
    cv::Mat img_blur;
    for (int i = FPS; i > 0; --i)
    {
        cv::GaussianBlur(img, img_blur, cv::Size(2 * i - 1, 2 * i - 1), 0);
        writer<<img_blur;
    }
}

//擦除
void transition_effect4(cv::VideoWriter& writer, const cv::Mat& img)
{
    cv::Mat img_map(img.size(), img.type());
    cv::Mat img_masked(img.size(), img.type());
    cv::Mat mask1(img.size(), img.type());
    cv::Mat mask2(img.size(), img.type());
    cv::Mat img_tmp(img.size(), img.type());
    
    for (int i = 0; i < FPS; ++i)
    {
        int interval = (img_map.cols - i * img_map.cols / FPS) / 255;
        cv::Mat tmp;
        for (int j = 0; j < 255; ++j)
        {
            tmp = img_map.colRange(j * interval, (j + 1) * interval);
            tmp.setTo(cv::Scalar(j, j, j));
        }
        tmp = img_map.colRange(255 * interval, img_map.cols);
        tmp.setTo(cv::Scalar(255, 255, 255));
        mask1.setTo(cv::Scalar(0, 0, 0));
        mask2.setTo(cv::Scalar(1, 1, 1));
        tmp = mask1.colRange(mask1.cols - i * mask1.cols / FPS, mask1.cols);
        tmp.setTo(cv::Scalar(1, 1, 1));
        tmp = mask2.colRange(mask2.cols - i * mask2.cols / FPS, mask2.cols);
        tmp.setTo(cv::Scalar(0, 0, 0));

        cv::multiply(img, mask1, img_masked);
        cv::multiply(img_map, mask2, img_tmp);
        img_masked = img_masked + img_tmp;
        writer<<img_masked;
    }
}

//二值化
void transition_effect5(cv::VideoWriter& writer, const cv::Mat& img)
{
    cv::Mat img_threshold(img.size(), img.type());
    for (int i = 0; i < FPS; ++i)
    {
        threshold(img, img_threshold, i * 255 / FPS, 255, cv::THRESH_BINARY);
        writer<<img_threshold;
    }
}

