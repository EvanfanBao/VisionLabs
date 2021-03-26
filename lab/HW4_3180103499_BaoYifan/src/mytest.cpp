#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

//对于每个人-前面5张训练
//第1张到第9张作为识别(test)到时候查找的范围--即识别匹配的时候-仅仅从1-到9张选择
//第10张是被识别图片

using namespace std;
using namespace cv;

const int PEOPLENUM = 41;
const int IMGPERPERSON = 10;
int countnum = 1;
class mytest
{
private:
    int WIDTHRESIZED, HEIGHTRESIZED;   //缩放比例
    Mat A_T;
    string datasetDir;
    vector<string> fileNames;
    vector<Mat> dataset;
public:
    mytest(string modelPath, string datasetDirctory);
    void test(string filePath);
};

mytest::mytest(string modelPath, string datasetDirectory)
{
    datasetDir = datasetDirectory;
    ifstream in(modelPath);
    double energyPercent;
    int fileSize;
    //从模型读入基本参数
    in >> WIDTHRESIZED >> HEIGHTRESIZED >> energyPercent >> fileSize;
    A_T = Mat(Size(WIDTHRESIZED * HEIGHTRESIZED, WIDTHRESIZED * HEIGHTRESIZED * energyPercent*0.01), CV_64F);
    //A_T size: WIDTHRESIZED * HEIGHTRESIZED * energyPercent*0.01 x WIDTHRESIZED * HEIGHTRESIZED
    for (int i = 0; i < fileSize; i++)
    {
        string fileName;
        in >> fileName;
        fileNames.push_back(fileName);
    }
    //读入A_T矩阵
    for (int i = 0; i < A_T.rows; i++)
    {
        for (int j = 0; j < A_T.cols; j++)
        {
            in >> A_T.at<double>(i, j);
        }
    }

    //遍历数据集的所有文件
    //添加对应的文件名
    //文件名格式为: si/j.pgm
    //最终路径为: datasetDir/si/j.pgm
    string fileName;
    fileNames.clear();
    for (int i = 1; i <= PEOPLENUM; i++)
    {
        for(int j = 1; j <= IMGPERPERSON - 1; j++) //1-到9张图片-第十章不算
        {
            fileName = 's' + to_string(i) + "/" + to_string(j) + ".pgm";
            fileNames.push_back(fileName);
            Mat img, imgResized;   //缩放用于加快特征值计算
            cvtColor(imread(datasetDir + fileName), img, COLOR_BGR2GRAY);
            resize(img, imgResized, Size(WIDTHRESIZED, HEIGHTRESIZED));
            imgResized.reshape(0, WIDTHRESIZED * HEIGHTRESIZED).convertTo(imgResized, CV_64F);
            Mat Y_T = A_T * imgResized; //同样转换为新的基下的表示
            dataset.push_back(Y_T);
        }
    }
}

//读入测试图片
//识别-找到最相似图片-输出
//同时将识别结果叠加在输入的人脸图像上
void mytest::test(string filePath)
{
    Mat imgTestOri, imgTest;

    imgTestOri = imread(filePath);
    cvtColor(imgTestOri, imgTest, COLOR_BGR2GRAY);
    resize(imgTest, imgTest, Size(WIDTHRESIZED, HEIGHTRESIZED));
    //转变为向量
    imgTest.reshape(0, WIDTHRESIZED * HEIGHTRESIZED).convertTo(imgTest, CV_64F);
    Mat mapped = A_T * imgTest;
    double min = -1;
    int index = 0;
    //计算欧式距离-找到最相近的图片
    for (int i = 0; i < dataset.size(); i++)
    {
        Mat diff = mapped - dataset[i];
        double mo = norm(diff);
        if(mo < min || min < 0)
        {
            min = mo;
            index = i;
        }
    }
    
    //输出相关结果
    cout << min;
    cout << "index is : " << index << endl;
    cout << "file name is : " << fileNames[index];
    Mat find = imread(datasetDir + fileNames[index]); //仅仅在前面用于制作特征向量的里面找-不合理-应该在所有的数据集里面找
    Mat blend;
    addWeighted(imgTestOri, 0.5, find, 0.4, 0, blend);
    
    imwrite("test_" + to_string(countnum) +".jpg", imgTestOri);
    imwrite("blend_" + to_string(countnum) + ".jpg", blend);
    imwrite("findMatch_" + to_string(countnum) + ".jpg", find);
    countnum++;
    namedWindow("blend Image");
    imshow("blend Image", blend);
    waitKey();
}

int main(int argc, char* argv[])
{
    if(argc !=4 )
    {
        cerr << "wrong argument" << endl;
        cerr << "./mytest filepath modelpath datasetdirectory";
    }
    string filePath = argv[1];  //图像输入文件名
    string modelPath = argv[2]; //图像模型名路径
    string datasetDir = argv[3];//数据集路径
    if('/' != *(datasetDir.end() -1) )
    {
        datasetDir.push_back('/');
    }
    mytest tes(modelPath, datasetDir);
    
    cout << "if your filePath is directory, please input 1(you are going to test a bunch of pictures and calc rank-1 rate)" << endl;
    int in;
    cout << "please input your choice" << endl;
    scanf("%d", &in);
    
    if(in == 1)
    {
        if('/' != *(filePath.end() -1) )
        {
            filePath.push_back('/');
        }
        cout << "begin looping";
        for (int k = 1; k <= 20; k++)
        {
            string fileName = filePath +  to_string(k) + ".pgm";
            tes.test(fileName);
        }
        
    }
    else
    {
        tes.test(filePath);
    }
    
    //mytest tes(modelPath, datasetDir);
    //tes.test(filePath);
    
}
