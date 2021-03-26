#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


class myreconstruct
{
private:
    int WIDTHRESIZED;
    int HEIGHTRESIZED;
    Mat A_T;
    
    vector<string> fileNames;
   
public:
    myreconstruct(string modelPath);
    void reconstruct(string filePath);
};

myreconstruct::myreconstruct(string modelPath)
{
   
    ifstream in(modelPath);
    double energyPercent;
    int fileSize;
    in >> WIDTHRESIZED >> HEIGHTRESIZED >> energyPercent >> fileSize;
    A_T = Mat(Size(WIDTHRESIZED * HEIGHTRESIZED, WIDTHRESIZED * HEIGHTRESIZED * energyPercent * 0.01), CV_64F);
    //读入文件名-此处无用-消耗掉内容信息
    for (int i = 0; i < fileSize; ++i)
    {
        string fileName;
        in>>fileName;
        fileNames.push_back(fileName);
    }
    //读入A_T 装载model 重要!
    for (int i = 0; i < A_T.rows; ++i)
    {
        for (int j = 0; j < A_T.cols; ++j)
        {
            in>>A_T.at<double>(i, j);
        }
    }
}

//重构
void myreconstruct::reconstruct(string filePath)
{
    Mat imgOri, img;
    imgOri = imread(filePath);
    cvtColor(imgOri, img, COLOR_BGR2GRAY);
    resize(img, img, Size(WIDTHRESIZED, HEIGHTRESIZED));
    img.reshape(0, WIDTHRESIZED * HEIGHTRESIZED).convertTo(img, CV_64F);
    Mat mapped = A_T * img; //映射yf = A_T * f
    //重构
    Mat A = Mat(Size(A_T.rows, A_T.cols), CV_64F);
    transpose(A_T, A);
    Mat f_hat = A * mapped; //重构fhat = A * y_f
    f_hat = f_hat.reshape(0, HEIGHTRESIZED);
    imwrite("reconstruction.jpg", f_hat);
}


int main(int argc, char* argv[])
{
    if(argc != 3 )
    {
        cerr << "wrong argument" << endl;
        cerr << "./myreconstruct modelpath datasetdirectory";
    }
    string filePath = argv[1];
    string modelPath = argv[2];
    myreconstruct rec(modelPath);
    rec.reconstruct(filePath);
}

