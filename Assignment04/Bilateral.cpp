#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <math.h>

#define IM_TYPE    CV_64FC3

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

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
Mat Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt);
Mat Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt);

int main() {

    Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
    Mat input_gray;

    // check for validation
    if (!input.data) {
        printf("Could not open\n");
        return -1;
    }

    cvtColor(input, input_gray, CV_RGB2GRAY);    // convert RGB to Grayscale
    
    // 8-bit unsigned char -> 64-bit floating point
    input.convertTo(input, CV_64FC3, 1.0 / 255);
    input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);

    // Add noise to original image
    Mat noise_Gray = Add_Gaussian_noise(input_gray, 0, 0.1);
    Mat noise_RGB = Add_Gaussian_noise(input, 0, 0.1);

    // Denoise, using gaussian filter
    Mat Denoised_Gray = Bilateralfilter_Gray(noise_Gray, 3, 10, 10, 10, "adjustkernel");
    Mat Denoised_RGB = Bilateralfilter_RGB(noise_RGB, 3, 10, 10, 10, "adjustkernel");

    namedWindow("Grayscale", WINDOW_AUTOSIZE);
    imshow("Grayscale", input_gray);

    namedWindow("RGB", WINDOW_AUTOSIZE);
    imshow("RGB", input);

    namedWindow("Gaussian Noise (Grayscale)", WINDOW_AUTOSIZE);
    imshow("Gaussian Noise (Grayscale)", noise_Gray);

    namedWindow("Gaussian Noise (RGB)", WINDOW_AUTOSIZE);
    imshow("Gaussian Noise (RGB)", noise_RGB);

    namedWindow("Denoised_Bilateral (Grayscale)", WINDOW_AUTOSIZE);
    imshow("Denoised_Bilateral (Grayscale)", Denoised_Gray);

    namedWindow("Denoised_Bilateral (RGB)", WINDOW_AUTOSIZE);
    imshow("Denoised_Bilateral (RGB)", Denoised_RGB);

    waitKey(0);

    return 0;
}

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma) {
                    //(input 이미지 행렬, 가우스 함수의 평균, 가우스 함수의 표준 편차)
    Mat NoiseArr = Mat::zeros(input.rows, input.cols, input.type());
    RNG rng;
    rng.fill(NoiseArr, RNG::NORMAL, mean, sigma);

    add(input, NoiseArr, NoiseArr);

    return NoiseArr;
}

Mat Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt) {
                    // (noise 있는 흑백 이미지, kernel 크기, x축에 대한 표준편차, y축에 대한 표준편차, 명도에 대한 표준편차, boundary의 픽셀 처리 방법)
    int row = input.rows;
    int col = input.cols;
    int kernel_size = (2 * n + 1);
    int tempx,tempy;


    Mat output = Mat::zeros(row, col, input.type());
    
    //w(s,t)중 input값이 필요없는 부분 먼저 계산
    Mat kernel1;
    kernel1 = Mat::zeros(kernel_size, kernel_size, CV_32F);
    for (int x = -n; x <= n; x++) {
        for (int y = -n; y <= n; y++) {
            kernel1.at<float>(x+n, y+n) = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
        }
    }
    
    Mat kernel2;
    
    // convolution
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {

            kernel2 = Mat::zeros(kernel_size, kernel_size, CV_32F);
            float sum = 0.0;
            
            if (!strcmp(opt, "zero-padding")) {
                float denom=0.0;    //W(i,j) : 모든 계산한 값의 합
                for (int x = -n; x <= n; x++) { // for each kernel window
                    for (int y = -n; y <= n; y++) {
                        if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
                            float value=exp(-(pow( (float)(input.at<G>(i, j))-(float)(input.at<G>(i+x, j+y)) ,2))/ (2*pow(sigma_r,2)));
                            float value2=kernel1.at<float>(x+n, y+n)*value;
                            kernel2.at<float>(x+n, y+n)=value2;     //각 픽셀에 해당하는 w(s,t)*W(i,j) (W(i,j)로 나누기 전) 값을 저장
                            denom += value2;
                        }

                    }
                }
                
                for (int x = -n; x <= n; x++) { // for each kernel window
                    for (int y = -n; y <= n; y++) {
                        if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
                            kernel2.at<float>(x+n, y+n)/=denom;                                 //W(i,j)로 나눠 w(s,t)를 구했다
                            sum+=kernel2.at<float>(x+n, y+n)*(float)(input.at<G>(i+x, j+y));    //가중치 w를 input값에 곱한 값들을 더한다
                        }

                    }
                }
                
                output.at<G>(i, j) = (G)sum;
            }

            else if (!strcmp(opt, "mirroring")) {
                float denom=0;
                for (int x = -n; x <= n; x++) { // for each kernel window
                    for (int y = -n; y <= n; y++) {
                        if (i + x > row - 1)    tempx = i - x;
                        else if (i + x < 0)     tempx = -(i + x);
                        else                    tempx = i + x;
                    
                        if (j + y > col - 1)    tempy = j - y;
                        else if (j + y < 0)     tempy = -(j + y);
                        else                    tempy = j + y;
                        
                        float value=exp(-1*(pow( (float)(input.at<G>(i, j))-(float)(input.at<G>(tempx, tempx)) ,2))/ (2*pow(sigma_r,2))) ;
                        float value2=kernel1.at<float>(x+n, y+n)*value;
                        kernel2.at<float>(x+n, y+n)=value2;
                        denom += value2;
                    }
                }
                for (int x = -n; x <= n; x++) { // for each kernel window
                    for (int y = -n; y <= n; y++) {
                        if (i + x > row - 1)    tempx = i - x;
                        else if (i + x < 0)     tempx = -(i + x);
                        else                    tempx = i + x;
                    
                        if (j + y > col - 1)    tempy = j - y;
                        else if (j + y < 0)     tempy = -(j + y);
                        else                    tempy = j + y;
                        
                        kernel2.at<float>(x+n, y+n)/=denom;
                        sum+=kernel2.at<float>(x+n, y+n)*(float)(input.at<G>(tempx, tempy));
                    }
                }
                output.at<G>(i, j) = (G)sum;
            }

            else if (!strcmp(opt, "adjustkernel")) {
                float sum2 = 0.0;
                float denom=0.0;
                for (int x = -n; x <= n; x++) { // for each kernel window
                    for (int y = -n; y <= n; y++) {
                        if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)){
                            float value=exp(-1*(pow( (float)(input.at<G>(i, j))-(float)(input.at<G>(i+x, j+y)) ,2))/ (2*pow(sigma_r,2))) ;
                            float value2=kernel1.at<float>(x+n, y+n)*value;
                            kernel2.at<float>(x+n, y+n)=value2;
                            denom += value2;
                        }
                    }
                }
                for (int x = -n; x <= n; x++) { // for each kernel window
                    for (int y = -n; y <= n; y++) {
                        if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)){
                            kernel2.at<float>(x+n, y+n)/=denom;
                            sum+=kernel2.at<float>(x+n, y+n)*(float)(input.at<G>(x+i, y+j));
                            sum2 += kernel2.at<float>(x+n, y+n);
                        }
                    }
                }
                output.at<G>(i, j) = (G)(sum/sum2);
            }

        }
    }

    return output;
}

Mat Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt) {
                    // (noise 있는 컬러 이미지, kernel 크기, x축에 대한 표준편차, y축에 대한 표준편차, 명도에 대한 표준편차, boundary의 픽셀 처리 방법)
    int row = input.rows;
    int col = input.cols;
    int kernel_size = (2 * n + 1);
    int tempx,tempy;


    Mat output = Mat::zeros(row, col, input.type());
    
    //w(s,t)중 input값이 필요없는 부분 먼저 계산
    Mat kernel1;
    kernel1 = Mat::zeros(kernel_size, kernel_size, CV_32F);
    for (int x = -n; x <= n; x++) {
        for (int y = -n; y <= n; y++) {
            kernel1.at<float>(x+n, y+n) = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
        }
    }
    
    Mat kernel2;
    kernel2 = Mat::zeros(kernel_size, kernel_size, CV_32F);

    // convolution
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            float sum_r = 0.0;
            float sum_g = 0.0;
            float sum_b = 0.0;

            if (!strcmp(opt, "zero-padding")) {
                float denom=0.0;        //W(i,j) : 모든 계산한 값의 합

                for (int x = -n; x <= n; x++) { // for each kernel window
                    for (int y = -n; y <= n; y++) {
                        if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
                            float Cp=0.0;
                            Cp+=pow((float)input.at<C>(i,j)[0]-(float)input.at<C>(i+x, j+y)[0],2);
                            Cp+=pow((float)input.at<C>(i,j)[1]-(float)input.at<C>(i+x, j+y)[1],2);
                            Cp+=pow((float)input.at<C>(i,j)[2]-(float)input.at<C>(i+x, j+y)[2],2);
                            Cp=sqrt(Cp);
                            //컬러 이미지는 R, G, B모두 고려한 Color Distance를 사용한다.
                            
                            float value=exp(-1*Cp/ (2*pow(sigma_r,2))) ;
                            float value2=kernel1.at<float>(x+n, y+n)*value;
                            kernel2.at<float>(x+n, y+n)=value2;
                            denom += value2;
                                
                        }
                    }
                }
                        
                for (int x = -n; x <= n; x++) { // for each kernel window
                    for (int y = -n; y <= n; y++) {
                        if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
                            kernel2.at<float>(x+n, y+n)/=denom;             //W(i,j)로 나눠 w(s,t)를 완성한다
                            sum_r+=kernel2.at<float>(x+n, y+n)*(float)(input.at<C>(x+i, y+j)[0]);
                            sum_g+=kernel2.at<float>(x+n, y+n)*(float)(input.at<C>(x+i, y+j)[1]);
                            sum_b+=kernel2.at<float>(x+n, y+n)*(float)(input.at<C>(x+i, y+j)[2]);
                            //완성된 가중치 값을 input 값에 곱한다
                        }
                    }
                }
                
                output.at<C>(i, j)[0] = (G)sum_r;
                output.at<C>(i, j)[1] = (G)sum_g;
                output.at<C>(i, j)[2] = (G)sum_b;
            }

            else if (!strcmp(opt, "mirroring")) {
            float denom=0.0;
            for (int x = -n; x <= n; x++) { // for each kernel window
                for (int y = -n; y <= n; y++) {
                    if (i + x > row - 1)    tempx = i - x;
                    else if (i + x < 0)     tempx = -(i + x);
                    else                    tempx = i + x;
                
                    if (j + y > col - 1)    tempy = j - y;
                    else if (j + y < 0)     tempy = -(j + y);
                    else                    tempy = j + y;
                    
                    float Cp=0.0;
                    Cp+=pow((float)input.at<C>(i,j)[0]-(float)input.at<C>(tempx, tempy)[0],2);
                    Cp+=pow((float)input.at<C>(i,j)[1]-(float)input.at<C>(tempx, tempy)[1],2);
                    Cp+=pow((float)input.at<C>(i,j)[2]-(float)input.at<C>(tempx, tempy)[2],2);
                    Cp=sqrt(Cp);
                    
                    float value=exp(-1*Cp/ (2*pow(sigma_r,2))) ;
                    float value2=kernel1.at<float>(x+n, y+n)*value;
                    kernel2.at<float>(x+n, y+n)=value2;
                    denom += value2;
                }
            }
            for (int x = -n; x <= n; x++) { // for each kernel window
                for (int y = -n; y <= n; y++) {
                    if (i + x > row - 1)    tempx = i - x;
                    else if (i + x < 0)     tempx = -(i + x);
                    else                    tempx = i + x;
                
                    if (j + y > col - 1)    tempy = j - y;
                    else if (j + y < 0)     tempy = -(j + y);
                    else                    tempy = j + y;
                        
                    kernel2.at<float>(x+n, y+n)/=denom;
                    sum_r+=kernel2.at<float>(x+n, y+n)*(float)(input.at<C>(tempx, tempy)[0]);
                    sum_g+=kernel2.at<float>(x+n, y+n)*(float)(input.at<C>(tempx, tempy)[1]);
                    sum_b+=kernel2.at<float>(x+n, y+n)*(float)(input.at<C>(tempx, tempy)[2]);
                    
                }
            }
            
            output.at<C>(i, j)[0] = (G)sum_r;
            output.at<C>(i, j)[1] = (G)sum_g;
            output.at<C>(i, j)[2] = (G)sum_b;
        }

        else if (!strcmp(opt, "adjustkernel")) {
            float sum2 = 0.0;
            float denom = 0.0;
            for (int x = -n; x <= n; x++) { // for each kernel window
                for (int y = -n; y <= n; y++) {
                    if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
                        float Cp=0.0;
                        Cp+=pow((float)input.at<C>(i,j)[0]-(float)input.at<C>(i+x, j+y)[0],2);
                        Cp+=pow((float)input.at<C>(i,j)[1]-(float)input.at<C>(i+x, j+y)[1],2);
                        Cp+=pow((float)input.at<C>(i,j)[2]-(float)input.at<C>(i+x, j+y)[2],2);
                        Cp=sqrt(Cp);
                        
                        float value=exp(-1*Cp/ (2*pow(sigma_r,2))) ;
                        float value2=kernel1.at<float>(x+n, y+n)*value;
                        kernel2.at<float>(x+n, y+n)=value2;
                        denom += value2;
                            
                            
                    }
                }
            }
            for (int x = -n; x <= n; x++) { // for each kernel window
                for (int y = -n; y <= n; y++) {
                    if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
                        
                        kernel2.at<float>(x+n, y+n)/=denom;
                        sum_r+=kernel2.at<float>(x+n, y+n)*(float)(input.at<C>(x+i, y+j)[0]);
                        sum_g+=kernel2.at<float>(x+n, y+n)*(float)(input.at<C>(x+i, y+j)[1]);
                        sum_b+=kernel2.at<float>(x+n, y+n)*(float)(input.at<C>(x+i, y+j)[2]);
                        sum2 += kernel2.at<float>(x+n, y+n);
                            
                    }
                }
            }
            output.at<C>(i, j)[0] = (G)(sum_r/sum2);
            output.at<C>(i, j)[1] = (G)(sum_g/sum2);
            output.at<C>(i, j)[2] = (G)(sum_b/sum2);
        }

    }
}

    return output;
}
