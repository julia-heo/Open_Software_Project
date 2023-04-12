#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3

using namespace cv;
float maxN=-1;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
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

Mat sobelfilter(const Mat input);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);    //"lena.jpg" 이미지를 컬러로 불러와서 input 변수에 대입
	Mat input_gray;
	Mat output;


	cvtColor(input, input_gray, CV_RGB2GRAY);   //input을 흑백으로 전환해 input_gray에 저장



	if (!input.data)
	{   //파일이 열리지 않았다면 "Could not open"출력 후 종료
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);  //흑백의 input 이미지 새로운 창으로 띄우기
	imshow("Grayscale", input_gray);
    
	output = sobelfilter(input_gray); //Boundary process: zero-paddle, mirroring, adjustkernel
    //output 행렬에 sobelfilter의 결과를 저장
    //매개변수: input이미지

	namedWindow("Sobel Filter", WINDOW_AUTOSIZE);
	imshow("Sobel Filter", output);

    printf("%f",maxN);  //이미지 색상중 최댓값 확인

	waitKey(0);

	return 0;
}


Mat sobelfilter(const Mat input) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N

	// Initialiazing 2 Kernel Matrix with 3x3 size for Sx and Sy
    Mat kernel_Sx = *(Mat_<float>(3, 3) << -1,0,1, -2,0,2, -1,0,1); //구체적으로 초기화
    Mat kernel_Sy = *(Mat_<float>(3, 3) << -1,-2,-1, 0,0,0, 1,2,1); //구체적으로 초기화
    //Fill code to initialize Sobel filter kernel matrix for Sx and Sy (Given in the lecture notes)
    //x,y방향으로 1차 편미분한 값을 근사화한것
    
	Mat output = Mat::zeros(row, col, input.type());

    int tempa;
    int tempb;
    
	for (int i = 0; i < row; i++) {                 //이미지의 모든 pixel에 대해
		for (int j = 0; j < col; j++) {
            float sumX = 0.0;
            float sumY = 0.0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( pow(input.at<G>(x, y)*Sx) + pow(input.at<G>(x, y)*Sy) )
                    //이미지의 범위(0~row-1, 0~col-1)를 벗어나면, kernel안에서 (i,j)를 기준으로 mirroring한 pixel의 값을 사용한다.
                    if (i + a > row - 1) tempa = i - a;
                    else if (i + a < 0)  tempa = -(i + a);
                    else    tempa = i + a;
            
                    if (j + b > col - 1)  tempb = j - b;
                    else if (j + b < 0)   tempb = -(j + b);
                    else    tempb = j + b;
                    
                    sumX += kernel_Sx.at<float>(a+n, b+n)*(float)(input.at<G>(tempa, tempb));   //Ix=Sx*I
                    sumY += kernel_Sy.at<float>(a+n, b+n)*(float)(input.at<G>(tempa, tempb));   //Iy=Sy*I
				}
			}
            output.at<G>(i, j) = sqrt( sumX*sumX + sumY*sumY );     //Sobel filter output
            if(maxN<output.at<G>(i, j)) maxN=output.at<G>(i, j);    //가장 큰 색상 값 구하기
		}
	}
	return output;
}
