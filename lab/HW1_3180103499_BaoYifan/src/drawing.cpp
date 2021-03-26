
#include "drawing.hpp"

using namespace cv;
using namespace std;
#define NUM_FRAME 300
#define SIZE 5

//便于实现的几个全局变量
VideoWriter writer;   //视频写流
Mat image, temp;
Point s;
char path[100];       //文件路径
//到中心位置
void toCenter(Point &a)
{
    a.x = a.x + WIDTH/2;
    a.y = -a.y + HEIGHT/2;
}
void init(VideoWriter& theWriter, char* thePath)
{
    image = Mat(HEIGHT, WIDTH, CV_8UC3);
    temp = Mat(HEIGHT, WIDTH, CV_8UC3);
    strcpy(path, thePath);      //文件路径
    writer = theWriter;         //视频写流
}

//缓慢绘制直线
void drawLine(Mat mat, Point start, Point end, Scalar color, int thick)
{
    int x1, y1, step;
    toCenter(start);
    toCenter(end);
    Point center = start;
    x1 = start.x;
    y1 = start.y;
    double footx, footy;
    int dx = end.x - start.x;
    int dy = end.y - start.y;
    circle(mat, center, thick, color, -1);
    writer.write(mat);
    abs(dx) > abs(dy) ? step = abs(dx) : step = abs(dy);
    footx = (double)dx / step;
    footy = (double)dy / step;
    for(int i = 0; i < step; i++)
    {
        footx > 0 ? x1 += int(footx+0.5) : x1 += int(footx - 0.5);
        footy > 0 ? y1 += int(footy + 0.5) : y1 += int(footy - 0.5);
        center.x = x1;
        center.y = y1;
        circle(mat, center, thick, color, -1);
        writer.write(mat);
    }
    center.x = center.x - WIDTH/2;
    center.y = -center.y + HEIGHT/2;
}

//背景图绘制
void drawBackground(bool flag)
{
    rectangle(image, Point(0, 0), Point(WIDTH, WIDTH), Scalar(255, 255, 255), -1, 8);
    if(flag == true)
    {
        writer.write(image);
    }
    rectangle(image, Point(0, 550), Point(WIDTH, WIDTH), Scalar(159, 200, 0), -1, 8);
}

void drawText()
{
    int i = 0;
    int count = 30;
    string name = "name : Bao Yifan";
    string number = "Student ID: 3180103499";
    string other = "OpenCV is powerful";
    while(1)
    {
        i++;
        putText(image, number, Point(WIDTH / 3, HEIGHT-i*(HEIGHT/3)/count), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0));
        putText(image, name, Point(WIDTH/3, 0 + i*(HEIGHT / 3)/count), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
        writer.write(image);
        if(i == count)
        {
            image.copyTo(temp);
            for(int j = 0; j <= count; j++)
            {
                temp.copyTo(image);
                putText(image, other, Point(75+j*(WIDTH/3)/count, HEIGHT/2), FONT_HERSHEY_COMPLEX, 1.5, Scalar(0, 255, 0));
                writer.write(image);
                if(j == count)
                {
                    //延迟一段时间
                    for (int q = 0; q < 200; q++)
                    {
                        writer.write(image);
                    }
                }
                drawBackground(false);
            }
            break;
        }
        else
        {
            drawBackground(false);  //重置image变量
        }
    }
    writer.write(image);
}

//故事情节 图中蕴含的道理1
void aphorism()
{
    int count = 30;
    string other = "You can not have it both ways";
    image.copyTo(temp);
    for(int j = 0; j <= count; j++)
    {
        temp.copyTo(image);
        putText(image, other, Point(j*(WIDTH/8)/count, HEIGHT/2), FONT_HERSHEY_COMPLEX, 1.5, Scalar(0, 255, 0));
        writer.write(image);
        if(j == count)
        {
            //延迟一段时间
            for (int i = 0; i < 200; i++)
            {
                writer.write(image);
            }
        }
        drawBackground(false);
    }
}

//故事情节 图中蕴含的道理2
void aphorism2()
{
    int count = 30;
    string other = "But it doesn't mean you are unhappy";
    string other2 = "Think small and do big";
    image.copyTo(temp);
    for(int j = 0; j <= count; j++)
    {
        temp.copyTo(image);
        putText(image, other, Point(j*(WIDTH/15)/count, HEIGHT/2), FONT_HERSHEY_COMPLEX, 1.5, Scalar(0, 255, 0));
        putText(image, other2, Point(j*(WIDTH/12)/count, HEIGHT/1.5), FONT_HERSHEY_COMPLEX, 1.5, Scalar(0, 255, 0));
        writer.write(image);
        if(j == count)
        {
            //延迟一段时间
            for (int i = 0; i < 200; i++)
            {
                writer.write(image);
            }
        }
        drawBackground(false);
    }
}

//缓慢绘制圆弧
void drawArc(Mat img, Point center, double radius, double start_angle, double end_angle, Scalar color, int thick)
{
    toCenter(center);
    Point arc;
    double foot = 0.02;
    for(double r  = start_angle; r <= end_angle; r += foot)
    {
        arc.x = center.x + radius * cos(r);
        arc.y = center.y + radius * sin(r);
        if(r == start_angle)
        {
            s = arc;
        }
        if(r == end_angle)
        {
            s = arc;
        }
        circle(img, arc, thick, color, -1);

        writer.write(image);
    }
}

//缓慢绘制椭圆的一部分
void drawEarc(Mat img, Point center, double start_angle, double end_angle, float a, float b, Scalar color, int thick, bool is_x)
{
    toCenter(center);
    Point arc;
    double foot = 0.02;
    for(double r = start_angle; r <= end_angle; r += foot)
    {
        if(is_x)
        {
            arc.x = center.x + a*cos(r);
            arc.y = center.y + b*sin(r);
        }
        else
        {
            arc.x = center.x + b*cos(r);
            arc.y = center.y + a*sin(r);
        }
        if(r == start_angle)
        {
            s = arc;
        }
        if(r == end_angle)
        {
            s = arc;
        }
        
        circle(img, arc, thick, color, -1);
        writer.write(image);
    }
}


void drawFish()
{
    Point eye(140, 160);
    toCenter(eye);
    Point bubbles[] = {Point(90, 140), Point(90, 155), Point(90, 170)};
    //鱼身子
    drawEarc(image, Point(200, 150), 0, (1 - 0.082) * PI, 100, 50, Scalar(0, 0, 0), 1, true);
    drawLine(image, Point(105, 140), Point(115, 150), Scalar(0, 0, 0), 1);
    drawLine(image, Point(115, 150), Point(105, 160), Scalar(0, 0, 0), 1);
    drawEarc(image, Point(200, 150), (1 + 0.082) * PI, 2 * PI, 100, 50, Scalar(0, 0, 0), 1, true);
    //鱼眼睛
    circle(image, eye, 5, Scalar(0, 0, 0), -1);
    //鱼脸部分割
    drawEarc(image, Point(125, 150), 0, 0.33 * PI, 35, 50, Scalar(0, 0, 0), 1, true);
    drawEarc(image, Point(125, 150), 1.67 * PI, 2 * PI, 35, 50, Scalar(0, 0, 0), 1, true);
    //鱼尾巴
    drawEarc(image, Point(340,150), 1.17 * PI, 1.5 * PI, 50, 25, Scalar(0, 0, 0), 1, true);
    drawEarc(image, Point(340, 150), PI, 1.5 * PI, 8, 25, Scalar(0,0,0), 1, true);
    drawEarc(image, Point(340, 150), 0.5 * PI, PI, 8, 25, Scalar(0,0,0), 1, true);
    drawEarc(image, Point(340,150), 0.5 * PI, 0.83 * PI, 50, 25, Scalar(0, 0, 0), 1, true);

    for(int i = 0; i < 3; i++)
    {
       drawEarc(image, bubbles[i], 0, 2 * PI, 5, 5, Scalar(0, 0, 0), 1, true);
    }
}

//绘制开心的表情
void drawWeirdFace()
{
    Point left_eye(-240, 160);
    Point right_eye(-180, 160);
    Point left_nose(-240,85);
    Point right_nose(-180,85);
    toCenter(left_eye);
    toCenter(right_eye);
    toCenter(left_nose);
    toCenter(right_nose);
    drawArc(image, Point(-210, 130), 100, 0, 2 * PI, Scalar(0, 0, 0), 1);
    //眼眶
    drawArc(image, Point(-170, 170), 20, 0, 2 * PI, Scalar(0, 0, 0), 1);
    drawArc(image, Point(-250, 170), 20, 0, 2 * PI, Scalar(0, 0, 0), 1);
    //眼球
    circle(image, left_eye, 6, Scalar(0, 0, 0), -1);
    writer.write(image);
    circle(image, right_eye, 6, Scalar(0, 0, 0), -1);
    writer.write(image);
    //鼻子
    drawEarc(image, Point(-210, 85), 0, PI, 60, 30, Scalar(0, 0, 0), 1, true);
}
//绘制爱心的表情
void drawLovingFace()
{
    drawArc(image, Point(210, 130), 100, 0, 2 * PI, Scalar(0, 0, 0), 1);
    //眼眶
    drawStar(image, Point(255, 180), 15, Scalar(0,0,0), 1);
    drawStar(image, Point(165, 180), 15, Scalar(0,0,0), 1);
    
    //笑 嘴巴张大
    drawEarc(image, Point(210, 110), 0, PI, 50, 50, Scalar(0,0,0), 1, true);
    drawEarc(image, Point(210, 60), 1.17 * PI, 1.83 * PI, 40, 40, Scalar(0,0,0), 1, true);
  
    drawLine(image, Point(210, 105), Point(260,125), Scalar(0,0,0), 1);
    drawLine(image, Point(210, 105), Point(160,125), Scalar(0,0,0), 1);
}

//绘制熊爪子
void drawBearPawl()
{
    //几个圆形的绘制
    drawEarc(image, Point(-170, 180), 0, 2 * PI, 10, 20, Scalar(0,0,0), 1, true);
    drawEarc(image, Point(-150, 220), 0, 2 * PI, 10, 20, Scalar(0,0,0), 1, true);
    drawEarc(image, Point(-120, 220), 0, 2 * PI, 10, 20, Scalar(0,0,0), 1, true);
    drawEarc(image, Point(-100, 180), 0, 2 * PI, 10, 20, Scalar(0,0,0), 1, true);
    Point center(-135, 170);
    toCenter(center);
    circle(image, center, 12, Scalar(0,0,0), 24);
}

//爱心形状绘制
void drawStar(Mat img,Point center,int a,Scalar color,int thick)
{
    toCenter(center);
    Point arc;
    double foot=2*PI/360;

    for (double i = -PI; i <= PI; i=i+foot)
    {
        arc.x = center.x + a*i*sin(PI*sin(i) / i);
        arc.y = center.y + a*abs(i)*cos(PI*sin(i) / i);
        circle(img, arc, thick, color, -1);
        writer.write(image);
    }
}
//绘制openCV的logo 作为最后的转场效果
void logo()
{
    // draw the red part
    for (int i = 0; i <= FPS; ++i)
    {
        cv::ellipse(image, cv::Point(WIDTH / 2, HEIGHT / 2 - 160), cv::Size(130, 130), 125, 0, i * 290 / FPS, CV_RGB(255, 0, 0), -1);
        cv::circle(image, cv::Point(WIDTH / 2, HEIGHT / 2 - 160), 60, CV_RGB(0, 0, 0), -1);
        writer<<image;
    }
    // draw the green part
    for (int i = 0; i <= FPS; ++i)
    {
        cv::ellipse(image, cv::Point(WIDTH / 2 - 150, HEIGHT / 2 + 80), cv::Size(130, 130), 16, 0, i * 290 / FPS, CV_RGB(0, 255, 0), -1);
        cv::circle(image, cv::Point(WIDTH / 2 - 150, HEIGHT / 2 + 80), 60, CV_RGB(0, 0, 0), -1);
        writer<<image;
    }
    // draw the blue part
    for (int i = 0; i <= FPS; ++i)
    {
        cv::ellipse(image, cv::Point(WIDTH / 2 + 150, HEIGHT / 2 + 80), cv::Size(130, 130), 300, 0, i * 290 / FPS, CV_RGB(0, 0, 255), -1);
        cv::circle(image, cv::Point(WIDTH / 2 + 150, HEIGHT / 2 + 80), 60, CV_RGB(0, 0, 0), -1);
        writer<<image;
      
    }
    for (int i = 0; i < FPS / 3; ++i)
    {
        writer<<image;
    }
}
//片尾动画
static Scalar randomColor(RNG& rng)
{
    int icolor = (unsigned)rng;
    return Scalar(icolor&255, (icolor>>8)&255, (icolor>>16)&255);
}
int Ending()
{
    const int NUMBER = 100;
    int lineType = LINE_AA; //抗锯齿线
    int i;
    //int width = 1000, height = 700;
    int x1 = -WIDTH/2, x2 = WIDTH*3/2, y1 = -HEIGHT/2, y2 = HEIGHT*3/2;
    RNG rng(0xFFFFFFFF);
    Mat image = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
    for (i = 0; i < NUMBER * 2; i++)
    {
        Point pt1, pt2;
        pt1.x = rng.uniform(x1, x2);
        pt1.y = rng.uniform(y1, y2);
        pt2.x = rng.uniform(x1, x2);
        pt2.y = rng.uniform(y1, y2);

        int arrowed = rng.uniform(0, 6);

        if( arrowed < 3 )
            line( image, pt1, pt2, randomColor(rng), rng.uniform(1,10), lineType );
        else
            arrowedLine(image, pt1, pt2, randomColor(rng), rng.uniform(1, 10), lineType);
        writer.write(image);
    }
    for (i = 0; i < NUMBER * 2; i++)
    {
        Point pt1, pt2;
        pt1.x = rng.uniform(x1, x2);
        pt1.y = rng.uniform(y1, y2);
        pt2.x = rng.uniform(x1, x2);
        pt2.y = rng.uniform(y1, y2);
        int thickness = rng.uniform(-3, 10);
        int marker = rng.uniform(0, 10);
        int marker_size = rng.uniform(30, 80);

        if (marker > 5)
            rectangle(image, pt1, pt2, randomColor(rng), MAX(thickness, -1), lineType);
        else
            drawMarker(image, pt1, randomColor(rng), marker, marker_size );

        writer.write(image);
    }
    for (i = 0; i < NUMBER; i++)
    {
        Point center;
        center.x = rng.uniform(x1, x2);
        center.y = rng.uniform(y1, y2);
        Size axes;
        axes.width = rng.uniform(0, 200);
        axes.height = rng.uniform(0, 200);
        double angle = rng.uniform(0, 180);

        ellipse( image, center, axes, angle, angle - 100, angle + 200,
                 randomColor(rng), rng.uniform(-1,9), lineType );

        writer.write(image);
    }
    for (i = 0; i< NUMBER; i++)
    {
        Point pt[2][3];
        pt[0][0].x = rng.uniform(x1, x2);
        pt[0][0].y = rng.uniform(y1, y2);
        pt[0][1].x = rng.uniform(x1, x2);
        pt[0][1].y = rng.uniform(y1, y2);
        pt[0][2].x = rng.uniform(x1, x2);
        pt[0][2].y = rng.uniform(y1, y2);
        pt[1][0].x = rng.uniform(x1, x2);
        pt[1][0].y = rng.uniform(y1, y2);
        pt[1][1].x = rng.uniform(x1, x2);
        pt[1][1].y = rng.uniform(y1, y2);
        pt[1][2].x = rng.uniform(x1, x2);
        pt[1][2].y = rng.uniform(y1, y2);
        const Point* ppt[2] = {pt[0], pt[1]};
        int npt[] = {3, 3};
        polylines(image, ppt, npt, 2, true, randomColor(rng), rng.uniform(1,10), lineType);
        writer.write(image);
    }
    for (i = 0; i< NUMBER; i++)
    {
        Point pt[2][3];
        pt[0][0].x = rng.uniform(x1, x2);
        pt[0][0].y = rng.uniform(y1, y2);
        pt[0][1].x = rng.uniform(x1, x2);
        pt[0][1].y = rng.uniform(y1, y2);
        pt[0][2].x = rng.uniform(x1, x2);
        pt[0][2].y = rng.uniform(y1, y2);
        pt[1][0].x = rng.uniform(x1, x2);
        pt[1][0].y = rng.uniform(y1, y2);
        pt[1][1].x = rng.uniform(x1, x2);
        pt[1][1].y = rng.uniform(y1, y2);
        pt[1][2].x = rng.uniform(x1, x2);
        pt[1][2].y = rng.uniform(y1, y2);
        const Point* ppt[2] = {pt[0], pt[1]};
        int npt[] = {3, 3};
        fillPoly(image, ppt, npt, 2, randomColor(rng), lineType);
        writer.write(image);
    }
    for (i = 0; i < NUMBER; i++)
    {
        Point center;
        center.x = rng.uniform(x1, x2);
        center.y = rng.uniform(y1, y2);
        circle(image, center, rng.uniform(0, 300), randomColor(rng),
               rng.uniform(-1, 9), lineType);
        writer.write(image);
    }

    for (i = 1; i < NUMBER; i++)
    {
        Point org;
        org.x = rng.uniform(x1, x2);
        org.y = rng.uniform(y1, y2);
        putText(image, "Bao Yifan, 3180103499", org, rng.uniform(0,8),
                rng.uniform(0,100)*0.05+0.1, randomColor(rng), rng.uniform(1, 10), lineType);
        writer.write(image);
    }
    Size textsize = getTextSize("End of this video", FONT_HERSHEY_COMPLEX, 3, 5, 0);
    Point org((WIDTH - textsize.width)/2, (HEIGHT - textsize.height)/2);
    Mat image2;
    for( i = 0; i < 255; i += 2 )
    {
        image2 = image - Scalar::all(i);
        putText(image2, "End of this video", org, FONT_HERSHEY_COMPLEX, 3,
                Scalar(i, i, 255), 5, lineType);
        writer.write(image2);
    }
    return 0;
}




