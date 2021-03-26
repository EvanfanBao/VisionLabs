

#ifndef Draw_hpp
#define Draw_hpp


#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "main.h"
#include <string>
#ifdef __APPLE__
           #include <sys/uio.h>
#else
           #include <sys/io.h>
#endif

#define PI 3.14

using namespace cv;

void toCenter(Point &a);
void init(VideoWriter& theWriter, char* thePath);
void play();
void drawLine(Mat mat, Point start, Point end, Scalar color, int thick);
void drawLine2(Mat mat, Point start, Point end, Scalar color, int thick);
void drawBackground(bool flag);
void drawText();
void drawArc(Mat img, Point center, double radius, double start_angle, double end_angle, Scalar color, int thick);
void drawEarc(Mat img, Point center, double start_angle, double end_angle, float a, float b, Scalar color, int thick, bool is_x);

void drawStar(Mat img,Point center,int a,Scalar color,int thick);


void drawFish();

void drawWeirdFace();
void drawLovingFace();
void drawBearPawl();
void aphorism();

void logo();


int Ending();
void testDrawing();
void aphorism2();

#endif /* Draw_hpp */
