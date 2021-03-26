#include "main.h"
using namespace std;
using namespace cv;
void myHoughLinePer(string imgPath, int threshold);
void myHoughCirclePer(string imgPath, int threshold);
int main(int argc, char** argv) {

    if(imgPath.empty())
    {
        //usage(argv[0]);
        return -1;
    }
    cv::namedWindow(CW_IMG_ORIGINAL, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(CW_IMG_EDGE,      cv::WINDOW_AUTOSIZE);
    cv::namedWindow(CW_ACCUMULATOR,     cv::WINDOW_AUTOSIZE);
    //myHoughLinePer(imgPath, thisThreshold);
    //myHoughCirclePer(imgPath, thisThreshold);
    myHoughLinePer(imgPath, thisThreshold);
    return 0;
}

void myHoughLinePer(string imgPath, int threshold)
{
    Mat gray;           //灰度图--用于边缘检测
    Mat imgCanny;       //边缘检测结果
    Mat imgBlur;        //平滑去噪音结果--用于边缘检测
    Mat imgOri =imread(imgPath, 1);  //原始图片读入
    cvtColor(imgOri, gray, COLOR_BGR2GRAY);
    blur(gray, imgBlur, Size(5, 5));  //平滑
    Canny(imgBlur, imgCanny, 100, 150, 3); //Canny 边缘检测 可以进行 参数调整
    int width = imgCanny.cols;          //图像宽度
    int height = imgCanny.rows;         //图像高度
    BaoYifan::HoughLine hough;
    hough.calAccum(imgCanny.data, width, height);
    if(threshold == 0)
        threshold = width>height?width/4:height/4; //默认threshold 后面可以根据按键进行修改 从而显示更多或者更少直线
    while(1)
    {
        Mat imgRes = imgOri.clone();    //最终结果图 结合 直线
        //遍历累加器 找大于threshhold的直线 返回直线的两个点对
        vector< pair< pair<int, int>, pair<int, int> > > lines = hough.getLines(threshold);//hough.GetLines(threshold);
        //根据点对 将直线绘制在结果图上
        vector< pair< pair<int, int>, pair<int, int> > >::iterator it;
        for(it=lines.begin();it!=lines.end();it++)
        {
            cv::line(imgRes, cv::Point(it->first.first, it->first.second), cv::Point(it->second.first, it->second.second), cv::Scalar( 0, 0, 255), 2, 8);   //直线绘制函数
        }
        //累加器的绘制
        int aw, ah, maxa;   //累加器的宽 高
        aw = ah = maxa = 0;
        const unsigned int* accu = hough.getAccum(&aw, &ah);   //返回对应累加器的数据
        //寻找最大计数值
        for(int p=0;p<(ah*aw);p++)
        {
            if((int)accu[p] > maxa)
                maxa = accu[p];
        }
        double contrast = 1.0;
        double coef = 255.0 / (double)maxa * contrast;  //以最大值为系数最大值
        cv::Mat img_accu(ah, aw, CV_8UC3);              //RGB图像
        for(int p=0;p<(ah*aw);p++)
        {
            unsigned char c = (double)accu[p] * coef < 255.0 ? (double)accu[p] * coef : 255.0;
            img_accu.data[(p*3)+0] = 255;       //RGB三个像素点的设置
            img_accu.data[(p*3)+1] = 255-c;
            img_accu.data[(p*3)+2] = 255-c;
        }
        cv::imshow(CW_IMG_ORIGINAL, imgRes);
        cv::imshow(CW_IMG_EDGE, imgCanny);
        cv::imshow(CW_ACCUMULATOR, img_accu);
        //按键响应 增大或者减小 threshold
        char c = cv::waitKey(360000);
        if(c == '+')
        {
            threshold += 5;
            cout << "print +" << endl;
        }
        if(c == '-')
        {
            threshold -= 5;
            cout << "print -" << endl;
        }
        if(c == 27)
            break;
    }
}

void myHoughCirclePer(string imgPath, int threshold)
{
   
    Mat gray;           //灰度图--用于边缘检测
    Mat imgCanny;       //边缘检测结果
    Mat imgBlur;        //平滑去噪音结果--用于边缘检测
    Mat imgOri =imread(imgPath, 1);  //原始图片读入
    cvtColor(imgOri, gray, COLOR_BGR2GRAY);
    blur(gray, imgBlur, Size(5, 5));  //平滑
    Canny(imgBlur, imgCanny, 100, 150, 3); //Canny 边缘检测 可以进行 参数调整
    int width = imgCanny.cols;          //图像宽度
    int height = imgCanny.rows;         //图像高度
    
    //
    BaoYifan::HoughCircle hough;
    vector< pair< pair<int, int>, int> > circles;
    for(int r =22; r < 100; r+=1)  //选择并遍历圆半径的大小范围
    {
        hough.calAccum(imgCanny.data, width, height, r);    //计算所有的投票
        if(threshold == 0)
            threshold = 1.01* (2.0 * (double)r * M_PI);    //默认threshold
                hough.getCircles(threshold, circles);       //遍历计数器 获取圆形的圆心
        
        //累加器的绘制
        int aw, ah, maxa;
        aw = ah = maxa = 0;
        const unsigned int* accu = hough.getAccum(&aw, &ah);

        for(int p=0;p<(ah*aw);p++)
        {
            if((int)accu[p] > maxa)
                maxa = accu[p];
        }
        double contrast = 1.0;
        double coef = 255.0 / (double)maxa * contrast;

        cv::Mat img_accu(ah, aw, CV_8UC3);
        for(int p=0;p<(ah*aw);p++)
        {
            unsigned char c = (double)accu[p] * coef < 255.0 ? (double)accu[p] * coef : 255.0;
            img_accu.data[(p*3)+0] = 255;
            img_accu.data[(p*3)+1] = 255-c;
            img_accu.data[(p*3)+2] = 255-c;
        }
        
        cv::imshow(CW_IMG_ORIGINAL, imgOri);
        cv::imshow(CW_IMG_EDGE, imgCanny);
        cv::imshow(CW_ACCUMULATOR, img_accu);
        waitKey(1);
    }

    int a,b,r;
    a=b=r=0;
    std::vector< std::pair< std::pair<int, int>, int> > result;
    std::vector< std::pair< std::pair<int, int>, int> >::iterator it;
    //前后不相交 适当的减少多余的圆//还可以进一步改进
    for(it=circles.begin();it!=circles.end();it++)
    {
        int d = sqrt( pow(abs(it->first.first - a), 2) + pow(abs(it->first.second - b), 2) );
        if( d > it->second + r)
        {
            result.push_back(*it);
            //ok
            a = it->first.first;
            b = it->first.second;
            r = it->second;
        }
        //result.push_back(*it);
    }
    //显示最终结果图
    Mat imgRes = imgOri.clone();
    for(it=result.begin();it!=result.end();it++)
    {
        circle(imgRes, cv::Point(it->first.first, it->first.second), it->second, cv::Scalar( 0, 255, 255), 2, 8);
    }
    imshow(CW_IMG_ORIGINAL, imgRes);
    imshow(CW_IMG_EDGE, imgCanny);
    waitKey(360000);
}
