
/* main.cpp
 author : Bao Yifan
 SID    : 3180103499
 e-mail : 3180103499@zju.edu.cn
 description: 视频绘制入口, 同时包含视频开头部分 即 开头视频+图片展示(浙大元素)
 */

#include "main.h"
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        cout << "invalid argument!" << endl << "./program --create: create video"
        << "./program --play: play video";
    }
    if(strcmp(argv[1],"--create") == 0){
        string path = "../lab1_pic/";     //视频和图片存储路径
        vector<Mat> pictures;                      //图片向量 用于保存读入图片
        string beginVideoPath = "../lab1_pic/begin.mp4";      //开场视频路径
        char writerPath[50] = "../generated_video/MySmallVideo.mp4"; //绘制视频写路径
        //读入图片并判断是否成功
        if(!getPictures(path, pictures))
        {
            cout << "read picture failed, exit";
            return 1;
        }
        namedWindow("Video");

        //VideoWriter 视频写类
        VideoWriter writer(writerPath, VideoWriter::fourcc('m', 'p', '4', 'v'), FPS, Size(WIDTH,HEIGHT));

        writeBeginVideo(writer, beginVideoPath);  //写片头
        writePictures(writer, pictures);          //写视频并过渡

        //缓慢图画绘制
        init(writer,writerPath);
        drawBackground(true);
        drawText();
        drawFish();
        drawBearPawl();
        aphorism();
        drawBackground(false);
        drawWeirdFace();
        drawLovingFace();
        aphorism2();
        drawBackground(false);
        image = Mat::zeros(HEIGHT,WIDTH,CV_8UC3);
        writer<<image;
        logo();
        Ending();
        writer.release();
    }
    else if(strcmp(argv[1], "--play") == 0)
    {
        play();
    }
    else
        cout << "invalid argument!" << endl << "./program --create: create video"
        << "./program --play: play video";
}


void play()
{
    char videoPath[50];
    strcpy(videoPath,"../generated_video/MySmallVideo.mp4");
    VideoCapture capture(videoPath);
    if(!capture.isOpened())
        cout << "fail to open!" << endl;
    long totalFrameNumber = capture.get(CAP_PROP_FRAME_COUNT);
    cout << "total frame number is" << totalFrameNumber << endl;
    //设置开始帧
    long frameToStart = 0;
    capture.set(CAP_PROP_POS_FRAMES, frameToStart); ///这里是CAP_PROP_POS_FRAMES 好像就是设置帧读取的起始位置？这里从第0帧开始
    cout << "start from" << frameToStart << "read" << endl;
    double rate = capture.get(CAP_PROP_FPS);
    cout << "the fps is: " << rate << endl;
    Mat frame;
    namedWindow("Video");
    int delay = 1000 / rate;
    long currentFrame = frameToStart;
    while(1)
    {
        if(!capture.read(frame))
        {
            break;
            return;
        }
        //这里加滤波程序
        imshow("Extratec frame", frame);
        int c = waitKey(delay); //有延迟 所以显示会慢一些
        if(c == 32) //遇到键盘空格暂停
        {
            waitKey(0);
        }
        currentFrame++;
    }
    capture.release();
    waitKey(0);
}



//将图片写入视频
void writePictures(VideoWriter& writer, vector<Mat>& pictures)
{
    Mat img, imgResized;
    //遍历图片
    for(int i = 0; i < pictures.size(); i++)
    {
        img = pictures[i];
        resize(img, imgResized, Size(WIDTH, HEIGHT));   //图像缩放
        //最后一帧 增加个人信息, 姓名与学号
        if(i == pictures.size() - 1)
        {
            cv::putText(imgResized, "Bao Yifan 3180103499", cv::Point(200, 100), FONT_HERSHEY_PLAIN, 4.0, cv::Scalar(0, 0, 0));
        }
        //每个写入后停2s 即重复写图片
        for(int j = 0; j < 2 * FPS; j++)
        {
            writer.write(imgResized);
        }
        transition(writer, imgResized);  //调用转场函数进行视频过渡
    }
}

//将开场视频写入视频
void writeBeginVideo(VideoWriter& writer, string& beginVideoPath)
{
    Mat img, imgResized;
    VideoCapture capture(beginVideoPath);
    Mat zjuIcon = imread("/Users/evan/lab1_pic/zju.jpg");  //浙江大学图标读入
    Rect roi_rect = Rect(350, 400, zjuIcon.cols, zjuIcon.rows);
    while (true)
    {
        capture.read(img);
        if (img.empty())
        {
            break;
        }
        cv::resize(img, imgResized, cv::Size(WIDTH, HEIGHT));
       
        zjuIcon.copyTo(imgResized(roi_rect));
        writer.write(imgResized);
    }
    capture.release();
}

//从对应路径获取图片
bool getPictures(string& path, vector<Mat>& pictures)
{
    if(access(path.c_str(), R_OK))
    {
        cout << "Directory does not exits!\n";
        return false;
    }
    
    for(int i = 1; i < PIC_NUM; i++)
    {
        char buffer[50];
        sprintf(buffer, "%s%d.jpg",path.c_str(),i);
        cout << buffer << endl;
        Mat temp = imread(buffer);
        pictures.push_back(temp);
    }
    cout << pictures.size();
    return true;
}



