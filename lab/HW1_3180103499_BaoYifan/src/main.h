
#ifndef main_h
#define main_h
#include <math.h>
#include <iostream>
#include <iostream>
#include <cstring>
#include <vector>
#include <regex>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include "transition.hpp"
#include "drawing.hpp"
#include <string.h>

#define PIC_NUM 6
#define WIDTH 1080
#define HEIGHT 720
#define FPS 30

using namespace cv;
using namespace std;
extern Mat image;

bool getPictures(string& path, vector<Mat>& pictures);
void writeBeginVideo(VideoWriter& writer, string& beginVideoPath);
void writePictures(VideoWriter& writer, vector<Mat>& pictures);
void play();



#endif /* main_h */
