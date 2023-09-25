#include <iostream>
#include <opencv2/opencv.hpp>

#define IM_TYPE    CV_8UC3
using namespace cv;


#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

Mat Kmeans(const Mat input,int clusterCount, int attempts);
Mat KmeansPosition(const Mat input,int clusterCount, int attempts,float sigmaX, float sigmaY);

int main() {

    Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);

    if (!input.data)
    {
        std::cout << "Could not open" << std::endl;
        return -1;
    }
    

    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", input);

    
    // the intensity value is not normalized here (0~1). normalize both intensity and position when using them simultaneously.
    int clusterCount = 10;
    int attempts = 5;
    float sigmaX=(float)input.cols/255.0;
    float sigmaY=(float)input.rows/255.0;
    
    Mat Output=Kmeans(input, clusterCount, attempts);   // Intensity만 반영
    Mat OutputPosition=KmeansPosition(input, clusterCount, attempts,sigmaX,sigmaY); // Intensity와 Positions모두 반영(3-d)
    
    imshow("clustered image", Output);
    imshow("clustered image + Position", OutputPosition);
    
    waitKey(0);

    return 0;
}

Mat Kmeans(const Mat input,int clusterCount, int attempts){
    Mat labels;
    Mat centers;
    
    // Clustering is performed for each channel (RGB)
    Mat samples(input.rows * input.cols, 3, CV_32F);
    for (int y = 0; y < input.rows; y++){
        for (int x = 0; x < input.cols; x++){ // 데이터 일차원으로 R G B 각각 저장
            samples.at<float>(y + x*input.rows, 0) = (float)(input.at<C>(y, x)[0]);
            samples.at<float>(y + x*input.rows, 1) = (float)(input.at<C>(y, x)[1]);
            samples.at<float>(y + x*input.rows, 2) = (float)(input.at<C>(y, x)[2]);
        }
    }
    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);   // Clustering 진행

    Mat new_image(input.size(), input.type());  // 결과 이미지 행렬

    for (int y = 0; y < input.rows; y++){
        for (int x = 0; x < input.cols; x++)
        {
            //Fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
            int cluster_idx = labels.at<int>(y + x*input.rows, 0);
            
            new_image.at <C>(y, x)[0] = (G)centers.at<float>(cluster_idx, 0);
            new_image.at <C>(y, x)[1] = (G)centers.at<float>(cluster_idx, 1);
            new_image.at <C>(y, x)[2] = (G)centers.at<float>(cluster_idx, 2);

        }
    }

    return new_image;
    
    
}

Mat KmeansPosition(const Mat input,int clusterCount, int attempts,float sigmaX, float sigmaY){
    Mat labels;
    Mat centers;
    
    // Clustering is performed for each channel (RGB)
    Mat samples(input.rows * input.cols, 5, CV_32F);
    for (int y = 0; y < input.rows; y++){
            for (int x = 0; x < input.cols; x++){
                samples.at<float>(y * input.cols+x, 0) = (float)(input.at<C>(y, x)[0]);
                samples.at<float>(y * input.cols+x, 1) = (float)(input.at<C>(y, x)[1]);
                samples.at<float>(y * input.cols+x, 2) = (float)(input.at<C>(y, x)[2]);
                
                // normalize
                samples.at<float>(y * input.cols+x, 3) =(float)(y)/sigmaY;
                samples.at<float>(y * input.cols+x, 4) = (float)(x)/sigmaX;
            }
    }
    

    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);   // Clustering 진행

    Mat new_image(input.size(), input.type());

    for (int y = 0; y < input.rows; y++){
           for (int x = 0; x < input.cols; x++)
           {
               int cluster_idx = labels.at<int>(y * input.cols+x, 0);
               
               new_image.at <C>(y, x)[0] = centers.at<float>(cluster_idx, 0);
               new_image.at <C>(y, x)[1] = centers.at<float>(cluster_idx, 1);
               new_image.at <C>(y, x)[2] = centers.at<float>(cluster_idx, 2);
           }
    }
    return new_image;
}
