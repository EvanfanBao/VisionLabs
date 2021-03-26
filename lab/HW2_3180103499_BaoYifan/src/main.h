
#ifndef main_h
#define main_h

//opencv库
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
//语言库
#include <stdio.h>
#include <string>
#include <map>
#include <iostream>
#include "hough.hpp"

std::string imgPath = "/Users/evan/Library/Developer/Xcode/DerivedData/cvTest-ebynplkownzjsachtzimeuxhcifw/Build/Products/Debug/myHighway.jpg";
int thisThreshold = 0;      //默认的threshhold 直接在函数中调整即可
const char* CW_IMG_ORIGINAL     = "Hough Result";
const char* CW_IMG_EDGE        = "Edge Detection Result";
const char* CW_ACCUMULATOR      = "Accumulator";
void myHoughLinePer(string imgPath, int threshold);
void myHoughCirclePer(string imgPath, int threshold);


#endif /* main_h */
