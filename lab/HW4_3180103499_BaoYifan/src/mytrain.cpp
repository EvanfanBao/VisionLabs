#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int PEOPLENUM = 41;
const int IMGPERPERSON = 10;

const int WIDTHRESIZED = 60;
const int HEIGHTRESIZED = 60;

class mytrain
{
private:
    vector<Mat> dataset;
    vector<string> fileNames;
    Mat A_T;
    //目前不是完全确定Mat格式的矩阵的索引方式
    //是A_T.at<>(行, 列)
    //还是A_T.at<>(列, 行)
    //因为从图像的角度开-水平(列)是x轴 竖直(行)是y轴
    //但这里作为矩阵就让我疑惑
    double energyPercent;
public:
    mytrain(string datasetDir, int trainNum);
    void train(double energyPerc);
    void write(string modelName);
};

//加载数据集
mytrain::mytrain(string datasetDir, int trainNum)
{
    if(trainNum > IMGPERPERSON)
    {
        cerr << "train number per person should no more than image per person";
    }
    string fileName;
    //遍历数据集的所有文件
    //添加对应的文件名
    //文件名格式为: si/j.pgm
    //最终路径为: datasetDir/si/j.pgm
    for (int i = 1; i <= PEOPLENUM; i++)
    {
        for(int j = 1; j <= trainNum; j++)
        {
            fileName = 's' + to_string(i) + "/" + to_string(j) + ".pgm";
            fileNames.push_back(fileName);
            Mat gray, imgResized;   //缩放用于加快特征值计算
            cvtColor(imread(datasetDir + fileName), gray, COLOR_BGR2GRAY);
            resize(gray, imgResized, Size(WIDTHRESIZED, HEIGHTRESIZED));
            dataset.push_back(imgResized);
        }
    }
}

void mytrain::train(double energyPerc)
{
    energyPercent = energyPerc;
    Mat covMat, mean;   //协方差矩阵 均值向量
    Mat eigenValues, eigenVectors;
    //计算协方差矩阵
    calcCovarMatrix(dataset, covMat, mean, COVAR_NORMAL);
    //covMat size: WIDTHRESIZED * HEIGHTRESIZED

    //根据协方差矩阵 计算得到排序的特征值与特征向量
    //特征向量s作为矩阵的形式
    //应该是矩阵的每一行是一个特征向量？不确定
    eigen(covMat, eigenValues, eigenVectors);

    //根据能量百分比获得 前面一些特征值对应的特征向量
    A_T = eigenVectors.rowRange(0, (int)(eigenVectors.rows * energyPercent * 0.01));
    //A_T size:
    //WIDTHRESIZED * HEIGHTRESIZED * energeyPercent * 0.01 x WIDTHRESIZED * HEIGHTRESIZED

    //检验结果的正确性
    cout << "A_T column is:" << A_T.cols << endl;
    cout << "A_T row is: " << A_T.rows << endl;

    //输出前10个特征脸
    std::vector<cv::Mat> firstTen;  //前十个特征脸
    cv::Mat firstTenConcat; //前十个特征脸连接
    for (int i = 0; i < 10; ++i)
    {
        cv::Mat tmp(cv::Size(WIDTHRESIZED, HEIGHTRESIZED), CV_64F);
        cv::Mat tmp_int(cv::Size(WIDTHRESIZED, HEIGHTRESIZED), CV_8UC1);
        for (int j = 0; j < WIDTHRESIZED * HEIGHTRESIZED; ++j)
        {
            tmp.at<double>(j / WIDTHRESIZED, j % WIDTHRESIZED) = eigenVectors.at<double>(i, j);
        }
        cv::normalize(tmp, tmp, 255, 0, cv::NORM_MINMAX);
        tmp.convertTo(tmp_int, CV_8UC1);
        firstTen.push_back(tmp_int);
    }
    cv::hconcat(firstTen, firstTenConcat);
    cv::imwrite("aa.jpg", firstTenConcat);

    //计算平均脸并输出
    Mat aveFloat = Mat::zeros(HEIGHTRESIZED, WIDTHRESIZED, CV_64F);
    Mat ave_res = Mat::zeros(HEIGHTRESIZED, WIDTHRESIZED, CV_8UC1);

    int total = 10;       //需要平均的脸的总数
    vector<Mat> allFaces;  //所有的脸 浮点
    for (int i = 0; i < total; i++)
    {
        Mat tmp(Size(WIDTHRESIZED, HEIGHTRESIZED),CV_64F);
        //特征向量转换为特征脸
        for (int j = 0; j < WIDTHRESIZED * HEIGHTRESIZED; j++)
        {
            tmp.at<double>(j / WIDTHRESIZED, j%WIDTHRESIZED) = eigenVectors.at<double>(i, j);
        }
        //normalize(tmp, tmp, 255, 0, NORM_MINMAX);
        allFaces.push_back(tmp);
    }
    for (int i = 0; i < total; i++)
    {
        aveFloat = aveFloat + allFaces.at(i);
    }
    aveFloat = aveFloat / (double)total;
    normalize(aveFloat, aveFloat, 255, 0, NORM_MINMAX);
    aveFloat.convertTo(ave_res, CV_8UC1);
    imwrite("ave_res.jpg",ave_res); //输出特征脸

}


//将模型的参数输出
void mytrain::write(string modelName)
{
    ofstream out(modelName);
    //输出缩放的图像宽度 高度
    out << WIDTHRESIZED << ' ' << HEIGHTRESIZED << '\n';
    //输出能量百分比
    out << energyPercent << '\n';
    //输出数据的文件名
    out << fileNames.size() << '\n';
    for (const auto& it: fileNames)
    {
        out << it << '\n';
    }
    //输出A_T矩阵-重要！
    for (int i = 0; i < A_T.rows; i++)
    {
        for (int j = 0; j < A_T.cols; j++)
        {
            out << A_T.at<double>(i, j)<<' ';
        }
    }
    out << '\n';
}


int main(int argc, char* argv[])
{
    if(argc != 4)
    {
        cerr << "wrong argument" << endl;
        cerr << "./mytrain energypercent modelname datasetdirectory";
    }
    double energyPercent = atof(argv[1]);
    string modelName = argv[2];
    cout << modelName;
    string datasetDir = argv[3];
    if('/' != *(datasetDir.end() -1) )
    {
        datasetDir.push_back('/');
    }
    mytrain tr(datasetDir,5);
    tr.train(energyPercent);
    tr.write(modelName);
}
