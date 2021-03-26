#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

//AT&T 数据集 rank1 识别率
//前5张训练 后5张测试 测试范围仍然是前5张

using namespace std;
using namespace cv;

const int PEOPLENUM = 40;

int countnum = 1;

int cate_count[41] = {0}; //正确判定的计数

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
    void test(string testFileName, int cate_index = 0);
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
        for(int j = 1; j <=  5; j++) //这里遍历1到5张图
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
void mytest::test(string testFileName, int cate_index)
{
    Mat imgTestOri, imgTest;

    imgTestOri = imread("./att_faces_with_eyes/" + testFileName);
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
    cout << "file name is : " << fileNames[index] << endl;
    string find_category = "";
    string test_category = "";
    
   
    for (int i = 0; i < fileNames[index].length(); i++)
    {
        if(fileNames[index][i] != '/' )
        {
            find_category.push_back(fileNames[index][i]);
            
        }
        else break;
    }
    //./att/s1/
    for (int i = 0; i < testFileName.length(); i++)
    {
        if(testFileName[i] != '/' )
        {
            test_category.push_back(testFileName[i]);
        }
        else break;
    }
        
    cout << "test_category is: " << test_category << endl;
    cout << "find_category is: " << find_category << endl;
    
    if(find_category == test_category)
    {
        cate_count[cate_index]++; ///判定正确--就+1---最多到5--因为一个目录就需要判定5张图片
    }
    Mat find = imread(datasetDir + fileNames[index]);
   
}


//AT&T数据集 rank-1测试
//内容写死-无命令行输入
int main(int argc, char* argv[])
{
    mytest tes("./model_7PCA", "./att_faces_with_eyes/");
    string fileName;
    
    for (int i = 1; i <= PEOPLENUM; i++)
    {
        for(int j = 6; j <= 10; j++)    //6-10 pic
        {
            fileName = 's' + to_string(i) + "/" + to_string(j) + ".pgm";
            //对每个图片进行测试即可
            cout << fileName;
            tes.test(fileName, i); //目录号就是i
        }
    }
    
    //遍历count-计算 rank-1 rate
    for(int i = 1; i <=40; i++ )
    {
        double rank_1_rate = cate_count[i] / 5.0;
        cout << "category " + to_string(i) + "ranke-1 rate: " << rank_1_rate << endl;
    }
}
