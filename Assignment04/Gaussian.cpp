#include <opencv2/opencv.hpp>
#include <stdio.h>

#define IM_TYPE	CV_64FC3

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
Mat Gaussianfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, const char *opt);
Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, const char *opt);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);    // 이미지 파일 가져와 컬러로 저장
	Mat input_gray;

	// check for validation
	if (!input.data) {          //파일 불러오기 실패하면 "Could not open\n"출력 후 종료
		printf("Could not open\n");
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale: 흑백으로 전환
	
	// 8-bit unsigned char -> 64-bit floating point
	input.convertTo(input, CV_64FC3, 1.0 / 255);
	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);

	// Add noise to original image
	Mat noise_Gray = Add_Gaussian_noise(input_gray, 0, 0.1);        //흑백 이미지에 Gaussian noise 추가
	Mat noise_RGB = Add_Gaussian_noise(input, 0, 0.1);              //컬러 이미지에 Gaussian noise 추가

	// Denoise, using gaussian filter
	Mat Denoised_Gray = Gaussianfilter_Gray(noise_Gray, 3, 5, 5, "adjustkernel");   // Gaussian noise 를 Gaussian Filter로 제거
	Mat Denoised_RGB = Gaussianfilter_RGB(noise_RGB, 3, 5, 5, "adjustkernel");

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Gaussian Noise (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (Grayscale)", noise_Gray);

	namedWindow("Gaussian Noise (RGB)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (RGB)", noise_RGB);

	namedWindow("Denoised (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Denoised (Grayscale)", Denoised_Gray);

	namedWindow("Denoised (RGB)", WINDOW_AUTOSIZE);
	imshow("Denoised (RGB)", Denoised_RGB);

	waitKey(0);

	return 0;
}

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma) {
                    //(input 이미지 행렬, 가우스 함수의 평균, 가우스 함수의 표준 편차)
	Mat NoiseArr = Mat::zeros(input.rows, input.cols, input.type());    // noise저장할 행렬
	RNG rng;
	rng.fill(NoiseArr, RNG::NORMAL, mean, sigma);                       // NoiseArr을 mean을 평균, sigma를 표준편차로 하는 가우스 함수의 수 중 랜덤으로 채운다.

	add(input, NoiseArr, NoiseArr);                                     // input 행렬과 NoiseArr 행렬을 더해 NoiseArr에 저장

	return NoiseArr;            // Noise추가한 이미지를 반환
}

Mat Gaussianfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, const char *opt) {
                    // (noise 있는 흑백 이미지, kernel 크기, x축에 대한 표준편차, y축에 대한 표준편차, boundary의 픽셀 처리 방법)
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
    int tempx,tempy;

	// Initialiazing Gaussian Kernel Matrix
	// Fill code to initialize Gaussian filter kernel matrix

	Mat output = Mat::zeros(row, col, input.type());
    

    //w(s,t)구하기
    //Zero mean Gaussian filter로 normalized한 식
    Mat kernel;
    kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);
    float denom = 0.0;
    for (int x = -n; x <= n; x++) {
        for (int y = -n; y <= n; y++) {
            float value = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
            kernel.at<float>(x+n, y+n) = value;
            denom += value;
        }
    }
    //전체 value들의 합으로 모든 픽셀을 나눠준다
    for (int x = -n; x <= n; x++) {
        for (int y = -n; y <= n; y++) {
            kernel.at<float>(x+n, y+n) /= denom;
        }
    }
    
    
	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
            float sum = 0.0;
            
			if (!strcmp(opt, "zero-padding")) {
				
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
                        if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
                            sum += kernel.at<float>(x+n, y+n)*(float)(input.at<G>(i+x, j+y));
                            //해당 픽셀의 이미지 데이터에 filtering의 가중치를 곱한 값을 sum에 더함
                        }

					}
				}
                output.at<G>(i, j) = (G)sum;    //필터링해 얻은 값을 (i,j)에 넣어준다
			}

			else if (!strcmp(opt, "mirroring")) {
				
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
                        //이미지의 범위(0~row-1, 0~col-1)를 벗어나면, kernel안에서 (i,j)를 기준으로 mirroring한 pixel의 값을 사용한다.
                        if (i + x > row - 1)    tempx = i - x;
                        else if (i + x < 0)     tempx = -(i + x);
                        else                    tempx = i + x;
                    
                        if (j + y > col - 1)    tempy = j - y;
                        else if (j + y < 0)     tempy = -(j + y);
                        else                    tempy = j + y;
                        
                        sum += kernel.at<float>(x+n, y+n)*(float)(input.at<G>(tempx, tempy));
                        //해당 픽셀의 이미지 데이터에 filtering의 가중치를 곱한 값을 sum에 더함
					}
				}
                output.at<G>(i, j) = (G)sum;    //필터링해 얻은 값을 (i,j)에 넣어준다
			}

			else if (!strcmp(opt, "adjustkernel")) {
                float sum2 = 0.0;   //kernel에 해당하는 픽셀의 비율을 더하는 변수
                
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
                        if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)){
                            sum += kernel.at<float>(x+n, y+n)*(float)(input.at<G>(i+x, j+y));   //해당 픽셀의 이미지 데이터에 filtering의 가중치를 곱한 값을 sum에 더함
                            sum2 += kernel.at<float>(x+n, y+n);     //이미지 범위 안에 들어가는 pixel의 가중치만을 더한다.
                        }
					}
				}
                output.at<G>(i, j) = (G)(sum/sum2);     //계산에 반영된 pixe들의 가중치 합으로 나눠준다. 이미지 안에 포함되는 kernel안의 pixel들만의 평균울 구할 수 있다.
			}

		}
	}

	return output;
}

Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, const char *opt) {
                    // (noise 있는 컬러 이미지, kernel 크기, x축에 대한 표준편차, y축에 대한 표준편차, boundary의 픽셀 처리 방법)
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
    int tempx,tempy;

	// Initialiazing Gaussian Kernel Matrix
	// Fill code to initialize Gaussian filter kernel matrix

	Mat output = Mat::zeros(row, col, input.type());
    Mat kernel;
    kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);
    
    //w(s,t)
    //Zero mean Gaussian filter로 normalized한 식
    float denom = 0.0;
    for (int x = -n; x <= n; x++) {
        for (int y = -n; y <= n; y++) {
            float value = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
            kernel.at<float>(x+n, y+n) = value;
            denom += value;
        }
    }
    //전체 value들의 합으로 모든 픽셀을 나눠준다
    for (int x = -n; x <= n; x++) {
        for (int y = -n; y <= n; y++) {
            kernel.at<float>(x+n, y+n) /= denom;
        }
    }
    

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
            //각 r,g, b의 값에 대해 따로따로 계산한다
            float sum_r = 0.0;
            float sum_g = 0.0;
            float sum_b = 0.0;

			if (!strcmp(opt, "zero-padding")) {
				
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

                        if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
                            sum_r += kernel.at<float>(x+n, y+n)*(float)(input.at<C>(i+x, j+y)[0]);
                            sum_g += kernel.at<float>(x+n, y+n)*(float)(input.at<C>(i+x, j+y)[1]);
                            sum_b += kernel.at<float>(x+n, y+n)*(float)(input.at<C>(i+x, j+y)[2]);
                            //해당 픽셀의 이미지 데이터에 filtering의 가중치를 곱한 값을 sum에 더함
                        }
					}
				}
                output.at<C>(i, j)[0] = (G)sum_r;
                output.at<C>(i, j)[1] = (G)sum_g;
                output.at<C>(i, j)[2] = (G)sum_b;
                //필터링해 얻은 값을 각 색의 (i,j)에 넣어준다
			}

			else if (!strcmp(opt, "mirroring")) {
				
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
                        if (i + x > row - 1)    tempx = i - x;
                        else if (i + x < 0)     tempx = -(i + x);
                        else                    tempx = i + x;
                    
                        if (j + y > col - 1)    tempy = j - y;
                        else if (j + y < 0)     tempy = -(j + y);
                        else                    tempy = j + y;
                        
                        sum_r += kernel.at<float>(x+n, y+n)*(float)(input.at<C>(i+x, j+y)[0]);
                        sum_g += kernel.at<float>(x+n, y+n)*(float)(input.at<C>(i+x, j+y)[1]);
                        sum_b += kernel.at<float>(x+n, y+n)*(float)(input.at<C>(i+x, j+y)[2]);
                        //해당 픽셀의 이미지 데이터에 filtering의 가중치를 곱한 값을 sum에 더함
					}
				}
                output.at<C>(i, j)[0] = (G)sum_r;
                output.at<C>(i, j)[1] = (G)sum_g;
                output.at<C>(i, j)[2] = (G)sum_b;
                //필터링해 얻은 값을 각 색의 (i,j)에 넣어준다
			}

			else if (!strcmp(opt, "adjustkernel")) {
                float sum2 = 0.0;
                for (int x = -n; x <= n; x++) { // for each kernel window
                    for (int y = -n; y <= n; y++) {
                        if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)){
                            sum_r += kernel.at<float>(x+n, y+n)*(float)(input.at<C>(i+x, j+y)[0]);
                            sum_g += kernel.at<float>(x+n, y+n)*(float)(input.at<C>(i+x, j+y)[1]);
                            sum_b += kernel.at<float>(x+n, y+n)*(float)(input.at<C>(i+x, j+y)[2]);
                                //해당 픽셀의 이미지 데이터에 filtering의 가중치를 곱한 값을 sum에 더함
                            sum2 += kernel.at<float>(x+n, y+n); //이미지 범위 안에 들어가는 pixel의 가중치만을 더한다.
                        }
                    }
                }
                output.at<C>(i, j)[0] = (G)(sum_r/sum2);
                output.at<C>(i, j)[1] = (G)(sum_g/sum2);
                output.at<C>(i, j)[2] = (G)(sum_b/sum2);
                //계산에 반영된 pixe들의 가중치 합으로 나눠준다. 이미지 안에 포함되는 kernel안의 pixel들만의 평균울 구할 수 있다.
			}

		}
	}

	return output;
}
