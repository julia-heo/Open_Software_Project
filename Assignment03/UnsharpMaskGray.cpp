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


Mat UnsharpMaskfilter (const Mat input, int n, float sigmaT, float sigmaS, const char* opt, float k);   //Unsharp Masking filter
Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);    //Low pass filter


int main() {
    
    Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);    //"lena.jpg" 컬러로 input에 대입
    Mat input_gray;
    Mat output;

    cvtColor(input, input_gray, CV_RGB2GRAY);   //input을 흑백으로 전환해 input_gray에 저장

    if (!input.data)
    {   //이미지 파일이 열리지 않았다면 "Could not open"출력 후 종료
        std::cout << "Could not open" << std::endl;
        return -1;
    }

    namedWindow("Grayscale", WINDOW_AUTOSIZE);
    imshow("Grayscale", input_gray);
    
    output = UnsharpMaskfilter(input_gray, 10, 7, 7, "mirroring", 0.3);
        //UnsharpMask(input, N, σt, σs, boundary_proc, k)
        //매개변수: input이미지, N (filter kernel (2N+1)x(2N+1)),x축에 대한 표준편차, y축에 대한 표준편차, type of boundary processing, losspass filter 반영 비율
    
    namedWindow("Unsharp Masking Filter", WINDOW_AUTOSIZE);
    imshow("Unsharp Masking Filter", output);


    waitKey(0);

    return 0;
}


Mat UnsharpMaskfilter (const Mat input, int n, float sigmaT, float sigmaS, const char* opt, float k) {
    int row = input.rows;
    int col = input.cols;
    Mat output = Mat::zeros(row, col, input.type());
    
    Mat lowpassfilter;
    lowpassfilter = gaussianfilter(input, n,sigmaT,sigmaS, opt);    //Low-pass filtering(gaussianfiltering)한 이미지 저장
    
    float temp;
    
    //Unsharp Masking의 output은 (I-kL)/(1-k)
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            for(int l=0;l<3;l++){
                temp=(input.at<G>(i, j) -  lowpassfilter.at<G>(i, j) * k)/(1 - k);
                
                if (temp < 0) temp = 0; //색 데이터가 0~255 벗어나는 경우 고정
                else if (temp > 255) temp = 255;
                
                output.at<G>(i, j) = (G)(temp);
            }
        }
    }
    
    return output;
}


Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {

    Mat kernel;
    
    int row = input.rows;
    int col = input.cols;
    int kernel_size = (2 * n + 1);
    int tempa;
    int tempb;
    float denom;

    // Initialiazing Kernel Matrix
    kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);   //kernel matrix 0으로 초기화
    
    
    denom = 0.0;        // Denominator in m(s,t)
    //fourier transform 만들기. w(s,t)구하기
    //Zero mean Gaussian filter로 normalized한 식
    for (int a = -n; a <= n; a++) {
        for (int b = -n; b <= n; b++) {
            float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))) - (pow(b, 2) / (2 * pow(sigmaT, 2))));
            kernel.at<float>(a+n, b+n) = value1;
            denom += value1;
        }
    }
    //전체 value들의 합으로 모든 픽셀을 나눠준다
    for (int a = -n; a <= n; a++) {
        for (int b = -n; b <= n; b++) {
            kernel.at<float>(a+n, b+n) /= denom;
        }
    }
    //(2*n+1)*(2*n+1) 크기의, 각 픽셀에 해당하는 가중치가 저장된 kernel 완성
    
    Mat output = Mat::zeros(row, col, input.type());
    
    
    for (int i = 0; i < row; i++) {         //이미지의 모든 pixel에 대해
        for (int j = 0; j < col; j++) {
            
            if (!strcmp(opt, "zero-paddle")) {      //Zero padding: 이미지의 boundary를 0으로 채운다
                float sum1 = 0.0;
                for (int a = -n; a <= n; a++) {
                    for (int b = -n; b <= n; b++) {
                                if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //이미지 범위 안의 픽셀이라면
                                    sum1 += kernel.at<float>(a+n, b+n)*(float)(input.at<G>(i + a, j + b));      //해당 픽셀의 이미지 데이터에 filtering의 가중치를 곱한 값을 sum1에 더함
                                }//이미지에서 벗어난 픽셀은 더해지지 않아 0으로 반영된다
                    }
                }
                output.at<G>(i, j) = (G)sum1;   //필터링해 얻은 값을 (i,j)에 넣어준다
            }
            
            else if (!strcmp(opt, "mirroring")) { //Mirroring: 이미지의 경계를 벗어난 픽셀은 반사된 값으로 boundary를 채운다
                float sum1 = 0.0;
                for (int a = -n; a <= n; a++) {
                    for (int b = -n; b <= n; b++) {

                        //mirroring for the border pixels
                        //이미지의 범위(0~row-1, 0~col-1)를 벗어나면, kernel안에서 (i,j)를 기준으로 mirroring한 pixel의 값을 사용한다.
                        if (i + a > row - 1) {      //kernel이 이미지의 오른쪽으로 삐져나간 경우
                            tempa = i - a;          //mirroring
                        }
                        else if (i + a < 0) {       //kernel이 이미지의 왼쪽으로 삐져나간 경우
                            tempa = -(i + a);       //mirroring
                        }
                        else {                      //kernel이 이미지 안에 잘 위치한 경우
                            tempa = i + a;
                        }
                        if (j + b > col - 1) {      //kernel이 이미지의 이래쪽으로 삐져나간 경우
                            tempb = j - b;          //mirroring
                        }
                        else if (j + b < 0) {       //kernel이 이미지의 위쪽으로 삐져나간 경우
                            tempb = -(j + b);       //mirroring
                        }
                        else {                      //kernel이 이미지 안에 잘 위치한 경우
                            tempb = j + b;
                        }
                        sum1 += kernel.at<float>(a+n, b+n)*(float)(input.at<G>(tempa, tempb)); //kernel 안의 모든 픽셀에 대해 가중치를 곱해 sum1에 더한다
                    }
                }
                output.at<G>(i, j) = (G)sum1;   //(i,j) 픽셀을 filtering한 값을 output의 (i,j)에 저장
            }


            else if (!strcmp(opt, "adjustkernel")) { //adjusting: 이미지 범위에 벗어나는 pixel은 아예 반영하지 않는다. kernel이 아미지의 범위를 벗어나면 kernel을 자른다
                
                float sum1 = 0.0;
                float sum2 = 0.0;   //kernel에 해당하는 픽셀의 비율을 더하는 변수
                
                for (int a = -n; a <= n; a++) { // for each kernel window
                    for (int b = -n; b <= n; b++) {
                        if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {     //이미지 범위에 벗어나지 않는다면
                            sum1 += kernel.at<float>(a+n, b+n)*(float)(input.at<G>(i + a, j + b));  //해당 픽셀의 이미지 데이터에 filtering의 가중치를 곱한 값을 sum1에 더함
                            sum2 += kernel.at<float>(a+n, b+n);    //이미지 범위 안에 들어가는 pixel의 가중치만을 더한다.
                        }
                    }
                }
                output.at<G>(i, j) = (G)(sum1/sum2);  //계산에 반영된 pixe들의 가중치 합으로 나눠준다. 이미지 안에 포함되는 kernel안의 pixel들만의 평균울 구할 수 있다.
            }
        }
    }
    return output;
}
