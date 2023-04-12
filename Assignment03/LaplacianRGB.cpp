#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE    CV_8UC3

using namespace cv;

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

float maxN=-1;

Mat laplacianfilter(const Mat input);

int main() {

    Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);    //"lena.jpg" 이미지를 컬러로 불러와서 input 변수에 대입

    Mat output;


    if (!input.data)
    {   //이미지 파일 열리지 않으면 "Could not open" 출력 후 종료
        std::cout << "Could not open" << std::endl;
        return -1;
    }

    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", input);
    output = laplacianfilter(input);    //매개변수: input이미지

    namedWindow("Laplacian Filter RGB", WINDOW_AUTOSIZE);
    imshow("Laplacian  Filter RGB", output);

    printf("%f\n",maxN);     //이미지 색상중 최댓값 확인

    waitKey(0);

    return 0;
}


Mat laplacianfilter(const Mat input) {
    
    int row = input.rows;
    int col = input.cols;
    int n = 1; // Sobel Filter Kernel N

    
    Mat kernel = *(Mat_<float>(3, 3) << 0,1,0, 1,-4,1, 0,1,0);//구체적으로 초기화
    //x,y방향으로 1차 편미분한 값을 근사화한것. Laplacian filter L

    Mat input_gray;
    cvtColor(input, input_gray, CV_RGB2GRAY); //ouput을 정의할 때, 흑백 이미지 type으로 설정하기 위해 input이미지를 흑백으로 전환
    Mat output = Mat::zeros(row, col, input_gray.type() );  //Laplacian filtering의 결과는 흑백
    
    int tempa;  //mirroring 반영 후의 픽셀 인덱스
    int tempb;
    
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            
            float sum_r = 0.0;
            float sum_g = 0.0;
            float sum_b = 0.0;
            
            for (int a = -n; a <= n; a++) {
                for (int b = -n; b <= n; b++) {
                    // Use mirroring boundary process
                    //이미지의 범위(0~row-1, 0~col-1)를 벗어나면, kernel안에서 (i,j)를 기준으로 mirroring한 pixel의 값을 사용한다.
                    if (i + a > row - 1) tempa = i - a;
                    else if (i + a < 0)  tempa = -(i + a);
                    else    tempa = i + a;
            
                    if (j + b > col - 1)  tempb = j - b;
                    else if (j + b < 0)   tempb = -(j + b);
                    else    tempb = j + b;
                    
                    //모든 kernel의 픽셀에 대해, 각각의 RGB에 대해 output=|L*I| 계산
                    sum_r += kernel.at<float>(a+n, b+n)*(float)(input.at<C>(tempa, tempb)[0]);
                    sum_g += kernel.at<float>(a+n, b+n)*(float)(input.at<C>(tempa, tempb)[1]);
                    sum_b += kernel.at<float>(a+n, b+n)*(float)(input.at<C>(tempa, tempb)[2]);
                    
                }
              
            }
            //절댓값, 뚜렷한 결과를 위해 *5
            sum_r = abs(sum_r)*5;
            sum_g= abs(sum_g)*5;
            sum_b = abs(sum_b)*5;
            //printf("%f %f %f\n",sum_r,sum_g,sum_b);
            //Laplacian filter output
            float out=(G)((sum_r+sum_g+sum_b)/3); //컬러 이미지는 laplacian filter의 결과M(x,y)는 세 채널의 laplacian filter 결과값의 평균
            if(out>255)out=255;         //색 범위를 벗어나는 경우 255로 고정
            output.at<G>(i, j) = out;   //Laplacian filter output
            if(maxN<output.at<G>(i, j)) maxN=output.at<G>(i, j);    //가장 큰 색상 값 구하기
        }
    }
    return output;
}
