#ifndef TRANSITION_H
#define TRANSITION_H

#include <opencv2/opencv.hpp>

#define WIDTH 1080
#define HEIGHT 720
#define FPS 30
void transition(cv::VideoWriter& writer, const cv::Mat& img);
void transition_effect1(cv::VideoWriter& writer, const cv::Mat& img);
void transition_effect2(cv::VideoWriter& writer, const cv::Mat& img);
void transition_effect3(cv::VideoWriter& writer, const cv::Mat& img);
void transition_effect4(cv::VideoWriter& writer, const cv::Mat& img);
void transition_effect5(cv::VideoWriter& writer, const cv::Mat& img);

typedef void transition_effect(cv::VideoWriter& writer, const cv::Mat& img);

#endif
