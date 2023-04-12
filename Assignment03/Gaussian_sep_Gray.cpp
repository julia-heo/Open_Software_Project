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

Mat Gaussianfilter_sep(const Mat input, int n, float sigmaT, float sigmaS, const char* opt); //Gaussian Filter 함수

int main() {
    
    Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);    //"lena.jpg" 컬러로 input에 대입
    Mat input_gray;
    Mat output;

    cvtColor(input, input_gray, CV_RGB2GRAY);               //input을 흑백으로 전환해 input_gray에 저장

    if (!input.data)
    {   //이미지 파일이 열리지 않았다면 "Could not open"출력 후 종료
        std::cout << "Could not open" << std::endl;
        return -1;
    }

    namedWindow("Grayscale", WINDOW_AUTOSIZE);
    imshow("Grayscale", input_gray);
    output = Gaussianfilter_sep(input_gray, 10,10,10, "zero-paddle");
        //Gaussianfilter_sep(input, N, σt, σs, boundary_proc)
        //매개변수: input이미지( gaussianfiltering 할 이미지), N (filter kernel (2N+1)x(2N+1)),x축에 대한 표준편차, y축에 대한 표준편차, type of boundary processing
    
    printf("%f",maxN);  //이미지 색상중 최댓값 확인
    
    for (int i = 0; i < input_gray.rows; i++) {         //색 데이터가 0~255에 골고루 퍼지게 하기 위해 maxN(가장 큰 색 데이터 값)로 나누고, 255를 곱한다
        for (int j = 0; j < input_gray.cols; j++) {
            output.at<G>(i, j) = (G)(output.at<G>(i, j)*255/maxN);
        }
    }
    
    namedWindow("Gaussian Filter", WINDOW_AUTOSIZE);    //결과 이미지 새로운 창으로 띄우기
    imshow("Gaussian Filter", output);

    waitKey(0);

    return 0;
}


Mat Gaussianfilter_sep(const Mat input, int n, float sigmaT, float sigmaS, const char* opt){

    int row = input.rows;
    int col = input.cols;
    int kernel_size = (2 * n + 1);
    int tempa;
    int tempb;
    float denom;
    
    //w(s,t)는 (2n+1)x(2n+1) 행렬이다. 이는 (2n+1)x1의 w_s(s,0)과 1x(2n+1)의 w_t(0,t)의 곱으로 나눌 수 있다
    //이 과정을 통해 계산량을 줄일 수 있다.

    // Initialiazing Kernel Matrix
    Mat kernel_s = Mat::zeros(kernel_size,1,CV_32F);    //w_s(s,0)
    Mat kernel_t = Mat::zeros(1,kernel_size,CV_32F);    //w_t(0,t)
    
    //kernel_s, w_s(s) 구하기
    denom = 0.0;    //분모
    for (int a = -n; a <= n; a++) {
            float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))));
            kernel_s.at<float>(a+n, 0) = value1;
            denom += value1;
    }
    //전체 value들의 합으로 모든 픽셀을 나눠준다
    for (int a = -n; a <= n; a++) {
            kernel_s.at<float>(a+n, 0) /= denom;
    }
    
    
    
    //kernel_t, w_t(t) 구하기
    denom = 0.0;    //분모
    for (int b = -n; b <= n; b++) {  // Denominator in m(s,t)
            float value1 = exp(-(pow(b, 2) / (2 * pow(sigmaT, 2))));
            kernel_t.at<float>(0, b+n) = value1;
            denom += value1;
    }
    //전체 value들의 합으로 모든 픽셀을 나눠준다
    for (int b = -n; b <= n; b++) {
            kernel_t.at<float>(0, b+n) /= denom;
    }

    
    Mat output = Mat::zeros(row, col, input.type());
    
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            //Gaussian filter with "zero-paddle" process
            if (!strcmp(opt, "zero-paddle")) {
                float sum1 = 0.0;
                Mat temp =Mat::zeros(kernel_size,1,CV_32F);

                for (int a = -n; a <= n; a++) {
                    for (int b = -n; b <= n; b++){
                        if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //이미지 범위 안의 픽셀이라면
                            temp.at<float>(a+n, 0)+=kernel_t.at<float>(0, b+n)*(float)(input.at<G>(i + a, j + b));
                                //1*(2*n+1) 크기의 kernel_t로 x축에 대한 계산 먼저 진행
                                //input(i+a,j-1), input(i+a,j), input(i+a,j+1)를 kernel_t으로 filtering하여 temp(a+n,0)에 임시 저장
                        }//이미지에서 벗어난 픽셀은 더해지지 않아 0으로 반영된다
                    }
                }

                
                for (int a = -n; a <= n; a++) {
                        sum1+=kernel_s.at<float>(a+n, 0)*temp.at<float>(a+n,0);
                        //(2*n+1)*1 크기의 kernel_s로 y축에 대한 계산 진행
                }
                output.at<G>(i, j) = (G)(sum1);
                if(maxN<output.at<G>(i, j)) maxN=output.at<G>(i, j);    //색 데이터의 최댓값을 찾는다
                
                
            }
            
            //Gaussian filter with "mirroring" process
            else if (!strcmp(opt, "mirroring")) {
                float sum1 = 0.0;
                Mat temp =Mat::zeros(kernel_size,1,CV_32F);
                for (int a = -n; a <= n; a++) {
                    for (int b = -n; b <= n; b++) {
                        //mirroring for the border pixels
                        //이미지의 범위(0~row-1, 0~col-1)를 벗어나면, kernel안에서 (i,j)를 기준으로 mirroring한 pixel의 값을 사용한다.
                        if (i + a > row - 1) tempa = i - a;
                        else if (i + a < 0)  tempa = -(i + a);
                        else    tempa = i + a;
                
                        if (j + b > col - 1)  tempb = j - b;
                        else if (j + b < 0)   tempb = -(j + b);
                        else    tempb = j + b;
                        
                        temp.at<float>(a+n, 0)+=kernel_t.at<float>(0, b+n)*(float)(input.at<G>(tempa, tempb));
                        //1*(2*n+1) 크기의 kernel_t로 x축에 대한 계산 먼저 진행
                        //input(i+a,j-1), input(i+a,j), input(i+a,j+1)를 kernel_t으로 filtering하여 temp(a+n,0)에 임시 저장
                                
                    }
                }
                for (int a = -n; a <= n; a++) {
                        sum1+=kernel_s.at<float>(a+n, 0)*temp.at<float>(a+n,0);
                    //(2*n+1)*1 크기의 kernel_s로 y축에 대한 계산 진행
                }
                output.at<G>(i, j) = (G)(sum1);
                if(maxN<output.at<G>(i, j)) maxN=output.at<G>(i, j); //색 데이터의 최댓값을 찾는다
            }

            // Gaussian filter with "adjustkernel" process:
            else if (!strcmp(opt, "adjustkernel")) {    //adjusting: 이미지 범위에 벗어나는 pixel은 아예 반영하지 않는다. kernel이 아미지의 범위를 벗어나면 kernel을 자른다
                float sum1 = 0.0;
                float sum2 = 0.0;   //kernel에 해당하는 픽셀의 비율을 더하는 변수
                Mat temp =Mat::zeros(kernel_size,1,CV_32F);
                Mat tempAdj =Mat::zeros(kernel_size,1,CV_32F);
                for (int a = -n; a <= n; a++) { // for each kernel window
                    for (int b = -n; b <= n; b++) {
                        if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
                            temp.at<float>(a+n, 0)+=kernel_t.at<float>(0, b+n)*(float)(input.at<G>(i + a, j + b));  //1*(2*n+1) 크기의 kernel_t로 x축에 대한 계산 먼저 진행. temp에 임시 저장
                            tempAdj.at<float>(a+n, 0)+=kernel_t.at<float>(0, b+n);//kernel_t 계산 시 사용된 가중치 더해 저장
                        }
                    }
                }
                for (int a = -n; a <= n; a++) {
                    sum1+=kernel_s.at<float>(a+n, 0)*temp.at<float>(a+n,0); //(2*n+1)*1 크기의 kernel_s로 y축에 대한 계산 진행
                    sum2 += kernel_s.at<float>(a+n, 0)*tempAdj.at<float>(0, a+n); //이미지 범위 안에 들어가는 pixel의 가중치만을 더한다. 두번에 걸쳐 계산하므로 계산된 픽셀의 가중치도 두번 계산 필요
                    
                }
                output.at<G>(i, j) = (G)(sum1/sum2);  //계산에 반영된 pixe들의 가중치 합으로 나눠준다. 이미지 안에 포함되는 kernel안의 pixel들만의 평균울 구할 수 있다.
                if(maxN<output.at<G>(i, j)) maxN=output.at<G>(i, j);    //색 데이터의 최댓값을 찾는다
            }
        }
    }
    return output;
}
