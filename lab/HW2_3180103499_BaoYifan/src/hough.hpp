
#ifndef HOUGH_H_
#define HOUGH_H_

#include <vector>
using namespace std;

namespace BaoYifan {
class HoughLine {
public:
    HoughLine();
    ~HoughLine();
public:
    int calAccum(unsigned char* imgData, int width, int height);
    vector< pair< pair<int, int>, pair<int, int> > > getLines(int threshold);
    const unsigned int* getAccum(int *width, int* height);
private:
    unsigned int* accumulator;
    int accumWidth;
    int accumHeight;
    int imgWidth;
    int imgHeight;
};

class HoughCircle {
public:
    HoughCircle();
    ~HoughCircle();
public:
    int calAccum(unsigned char* imgData, int width, int height, int r);
    int getCircles(int threshold, vector< pair< pair<int, int>, int> >& result);
    const unsigned int* getAccum(int *width, int *height);
private:
    unsigned int* accumulator;
    int accumWidth;
    int accumHeight;
    int imgWidth;
    int imgHeight;
    int radius;
};

}


#endif


