#include <iostream>
#include <opencv2/opencv.hpp>

#define IM_TYPE	CV_8UC3
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
    
    cvtColor(input, input, CV_RGB2GRAY);        // 흑백으로 전환

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);

    int clusterCount = 10;   //이 변수를 조정하여 결과 확인(군집화할 개수)
    int attempts = 5;
    
    // the intensity value is not normalized here (0~1). normalize both intensity and position when using them simultaneously.
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
    
    Mat samples(input.rows * input.cols, 1, CV_32F);
    for (int y = 0; y < input.rows; y++)
        for (int x = 0; x < input.cols; x++)
            samples.at<float>(y + x*input.rows, 0) = (float)(input.at<uchar>(y, x));    // 데이터 일차원으로 저장

    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

    Mat new_image(input.size(), input.type());  // 결과 이미지 행렬

    for (int y = 0; y < input.rows; y++){
        for (int x = 0; x < input.cols; x++)
        {
            //Fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
            int cluster_idx = labels.at<int>(y + x*input.rows, 0);
            new_image.at <uchar>(y, x) = (uchar)centers.at<float>(cluster_idx, 0);

        }
    }
    return new_image;
}

Mat KmeansPosition(const Mat input,int clusterCount, int attempts,float sigmaX, float sigmaY){
    Mat labels;
    Mat centers;
    
    Mat samples(input.rows * input.cols, 3, CV_32F);
    for (int y = 0; y < input.rows; y++){
            for (int x = 0; x < input.cols; x++){
                samples.at<float>(y * input.cols+x, 0) = (float)(input.at<G>(y, x));
                samples.at<float>(y * input.cols+x, 1) =(float)(y)/sigmaY;
                samples.at<float>(y * input.cols+x, 2) = (float)(x)/sigmaX;
            }
    }

    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

    Mat new_image(input.size(), input.type());

    for (int y = 0; y < input.rows; y++){
           for (int x = 0; x < input.cols; x++)
           {
               int cluster_idx = labels.at<int>(y * input.cols+x, 0);
               new_image.at<G>(y, x) = (G)((centers.at<float>(cluster_idx, 0)));
           }
    }
    return new_image;
}
