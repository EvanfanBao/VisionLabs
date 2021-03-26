
#include "hough.hpp"
#include <cmath>
#include <iostream>
#include <string.h>
#include <stdlib.h>


#define DEG2RAD 0.017453293f
using namespace std;


namespace BaoYifan {
HoughLine::HoughLine():accumulator(0),accumWidth(0),accumHeight(0),imgWidth(0),imgHeight(0){} //构造函数
HoughLine::~HoughLine(){if(accumulator)free(accumulator);} //析构函数

int HoughLine::calAccum(unsigned char* imgData, int width, int height)
{
    imgWidth = width;       //图像宽度
    imgHeight = height;     //图像高度
    //创建计数器
    double r_max = ((sqrt(2.0) * (double)(height>width?height:width)) / 2.0); //对角线 最长 r的最大值
    accumHeight = r_max * 2.0; //这里 -r到r 因此总共坐标长度为2r_max--0到2r_max对应到-r到r
    accumWidth = 180;          ///180弧度 横坐标
    accumulator = (unsigned int*)calloc(accumWidth*accumHeight, sizeof(unsigned int));  //二维数组
    double centerX = width / 2;  //图像中心点x坐标
    double centerY = height / 2; //图像中心点y坐标
      
    //遍历行列像素点 转换 设置累加器
    //IMPORTATN
    for(int py = 0; py < height; py++)
    {
        for(int px = 0; px < width; px++)
        {
            if(imgData[(py*width + px)] > 250)
            {   //遍历所有的theta(0-180)计算对应的r 即表示一个像素点 贡献的他经过的所有直线
                for(int theta = 0; theta < 180; theta++)
                {
                    double r = ( ((double)px - centerX) * cos((double)theta * DEG2RAD)) + (((double)py - centerY) * sin((double)theta * DEG2RAD));//公式r=xcos(theta)+ysin(theta)计算离图像中点的距离//有负有正
                    accumulator[ (int)((round(r + r_max) * 180.0)) + theta]++; //对应的累加器计数+1
                }
            }
        }
    }
    return 0;
}
vector< pair< pair<int, int>, pair<int, int> > > HoughLine:: getLines(int threshold)
{
    vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines; //返回直线结果
    if(accumulator==NULL)
        return lines;   //空
    //遍历accumulator
    for(int r = 0; r < accumHeight; r++)    //行r
    {
        for(int theta = 0; theta < accumWidth; theta++) //列theta
        {
            if((int)accumulator[(r*accumWidth)+theta] >= threshold) //判断大于 threshold 则此点/直线(r theta)满足
            {
                //判断是否是9x9格子中的局部最大
                int max = accumulator[(r*accumWidth) + theta];
                for(int ly=-4;ly<=4;ly++)
                {
                    for(int lx=-4;lx<=4;lx++)
                    {
                        if( (ly+r>=0 && ly+r< accumHeight) && (lx+theta>=0 && lx+theta
                                                               <accumWidth)  )
                        {
                            if( (int)accumulator[( (r+ly)*accumWidth) + (theta+lx)] > max )
                            {
                                max = accumulator[( (r+ly)*accumWidth) + (theta+lx)];
                                ly = lx = 5;
                            }
                        }
                    }
                }
                if(max > (int)accumulator[(r*accumWidth) + theta])
                    continue;
                
                //反计算 直线上两个点的值 并加入到直线中
                //公式 y = (r - xcos(theta)) / sin(theta)
                //公式 x = (r - ysin(theta)) / cos(theta)
                int x1, y1, x2, y2;
                x1 = y1 = x2 = y2 = 0;
                if(theta >= 45 && theta <= 135) //这里条件判断主要是为了避开两个等于0的情况
                {
                    x1 = 0; //第一个点的选取 选择y轴上点
                    y1 = ((double)(r-(accumHeight/2)) - ((x1 - (imgWidth/2) ) * cos(theta * DEG2RAD))) / sin(theta * DEG2RAD) + (imgHeight / 2);
                    x2 = imgWidth;  //第二个点的选取 选择与图像右边缘的交点
                    y2 = ((double)(r-(accumHeight/2)) - ((x2 - (imgWidth/2) ) * cos(theta * DEG2RAD))) / sin(theta * DEG2RAD) + (imgHeight / 2);
                }
                else
                {
                    y1 = 0;  //第一个点的选取 x轴上的点
                    x1 = ((double)(r-(accumHeight/2)) - ((y1 - (imgHeight/2) ) * sin(theta * DEG2RAD))) / cos(theta * DEG2RAD) + (imgWidth / 2);
                    y2 = imgHeight;    //第二个点的选取 选择与图像下边缘的交点
                    x2 = ((double)(r-(accumHeight/2)) - ((y2 - (imgHeight/2) ) * sin(theta * DEG2RAD))) / cos(theta * DEG2RAD) + (imgWidth / 2);
                    
                }
                lines.push_back(pair< pair<int, int>, pair<int, int> >(pair<int, int>(x1,y1), pair<int, int>(x2,y2)));
            }
        }
    }
    return lines;
}

//返回累加器二维数组 用于绘制图像
const unsigned int* HoughLine:: getAccum(int *width, int* height)
{
    *width = accumWidth;
    *height = accumHeight;
    return accumulator;
}


HoughCircle::HoughCircle():accumulator(NULL),accumWidth(0),accumHeight(0),imgWidth(0), imgHeight(0){}//构造函数
HoughCircle::~HoughCircle(){if(accumulator!=NULL)free(accumulator);}    //析构函数
int HoughCircle::calAccum(unsigned char *imgData, int width, int height, int r)
{
    radius = r;         //圆半径 //在外面遍历和设置
    imgWidth = width;   //宽度
    imgHeight = height; //高度
    //创建计数器
    accumWidth = width;
    accumHeight = height; //以圆心ab为坐标系 因此宽与高和图像一样大
    if(accumulator != NULL)
        free(accumulator); //原先有的 先情况
    accumulator = (unsigned int*)calloc(accumHeight * accumWidth, sizeof(unsigned int));
    for(int py = 0; py < height; py++)      //遍历行
    {
        for(int px = 0; px < width; px++)   //遍历列
        {
            if(imgData[py*width+px] > 250)
            {
                for(int theta=1; theta <= 360; theta++) //遍历所有可能的角度 即每个边缘点 贡献其所在的对应半径的所有圆(圆心计数器)
                {
                    //计算圆心
                    int a = ((double)px - ((double)radius * cos((double)theta * DEG2RAD)));
                    int b = ((double)py - ((double)radius * sin((double)theta * DEG2RAD)));
                    if( (b>=0 && b<accumHeight) && (a>=0 && a<accumWidth))  //没有超出对应的范围
                        accumulator[(b * accumWidth) + a]++;    //计数器+1
                }
            }
        }
    }
    return 0;
}
int HoughCircle::getCircles(int threshold, vector< pair< pair<int, int>, int> >& result)
{
    int count = 0;
    if(accumulator==NULL)
        return 0;
    for(int b = 0; b < accumHeight; b++)    //遍历行
    {
        for(int a = 0; a < accumWidth; a++) //遍历列
        {
            if((int)accumulator[b * accumWidth + a] >= threshold) //大于threshold才选择
            {
                //判断9x9的局部最大值
                int max = accumulator[(b * accumWidth) + a];
                for(int ly=-4;ly<=4;ly++)
                {
                    for(int lx=-4;lx<=4;lx++)
                    {
                        if( (ly+b>=0 && ly+b<accumHeight) && (lx+a>=0 && lx+a<accumWidth)  )
                        {
                            if( (int)accumulator[( (b+ly)*accumWidth) + (a+lx)] > max )
                            {
                                max = accumulator[( (b+ly)*accumWidth) + (a+lx)];
                                ly = lx = 5;
                            }
                        }
                    }
                }
                if(max > (int)accumulator[(b * accumWidth) + a])
                    continue;
                result.push_back(pair< pair<int, int>, int>(pair<int, int>(a,b), radius));  //加入向量
                count++;
            }
        }
    }
    return count;   //返回加入的个数
}

//用于绘制累加器
const unsigned int* HoughCircle::getAccum(int *width, int *height)
{
    *width = accumWidth;
    *height = accumHeight;

    return accumulator;
}

}



