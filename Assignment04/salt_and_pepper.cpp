#include <opencv2/opencv.hpp>
#include <stdio.h>

#define IM_TYPE	CV_8UC3

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

Mat Add_salt_pepper_Noise(const Mat input, float ps, float pp);
Mat Salt_pepper_noise_removal_Gray(const Mat input, int n, const char *opt);
Mat Salt_pepper_noise_removal_RGB(const Mat input, int n, const char *opt);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);    //이미지 파일 가져와 컬러로 저장
	Mat input_gray;

	// check for validation
	if (!input.data) {          //파일 불러오기 실패하면 "Could not open\n"출력 후 종료
		printf("Could not open\n");
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale: 흑백으로 전환하여 저장

	// Add noise to original image
	Mat noise_Gray = Add_salt_pepper_Noise(input_gray, 0.1f, 0.1f); //흑백 이미지에 salt ans pepper noise 추가
	Mat noise_RGB = Add_salt_pepper_Noise(input, 0.1f, 0.1f);       //컬러 이미지에 salt ans pepper noise 추가

	// Denoise, using median filter
	int window_radius = 2;
	Mat Denoised_Gray = Salt_pepper_noise_removal_Gray(noise_Gray, window_radius, "adjustkernel");  //median filter
	Mat Denoised_RGB = Salt_pepper_noise_removal_RGB(noise_RGB, window_radius, "adjustkernel");

	namedWindow("Grayscale", WINDOW_AUTOSIZE);  //흑백 이미지 띄우기
	imshow("Grayscale", input_gray);

	namedWindow("RGB", WINDOW_AUTOSIZE);        //컬러 이미지 띄우기
	imshow("RGB", input);

	namedWindow("Impulse Noise (Grayscale)", WINDOW_AUTOSIZE);      //salt and pepper noise 추가한 흑백 이미지 띄우기
	imshow("Impulse Noise (Grayscale)", noise_Gray);

	namedWindow("Impulse Noise (RGB)", WINDOW_AUTOSIZE);            //salt and pepper noise 추가한 컬러 이미지 띄우기
	imshow("Impulse Noise (RGB)", noise_RGB);

	namedWindow("Denoised (Grayscale)", WINDOW_AUTOSIZE);           //meanfiltering으로 salt and pepper noise 제거한 흑백 이미지 띄우기
	imshow("Denoised (Grayscale)", Denoised_Gray);

	namedWindow("Denoised (RGB)", WINDOW_AUTOSIZE);                 //meanfiltering으로 salt and pepper noise 제거한 컬러 이미지 띄우기
	imshow("Denoised (RGB)", Denoised_RGB);

	waitKey(0);

	return 0;
}

Mat Add_salt_pepper_Noise(const Mat input, float ps, float pp)
{                       //(input 이미지 행렬, salt noise 비율, pepper noise 비율)
    
	Mat output = input.clone();         // output에 input이미지 복사해 저장
	RNG rng;

	int amount1 = (int)(output.rows * output.cols * pp);    //salt noise 픽셀 수 계산
	int amount2 = (int)(output.rows * output.cols * ps);    //pepper noise 픽셀 수 계산

	int x, y;

	// Grayscale image
	if (output.channels() == 1) {   // 흑백 이미지
		for (int counter = 0; counter < amount1; ++counter)
			output.at<G>(rng.uniform(0, output.rows), rng.uniform(0, output.cols)) = 0;     //amount1개의 난수 발생시켜 0(흰색)으로 데이터 바꾼다

		for (int counter = 0; counter < amount2; ++counter)
			output.at<G>(rng.uniform(0, output.rows), rng.uniform(0, output.cols)) = 255;   //amount2개의 난수 발생시켜 255(검정색)으로 데이터 바꾼다
	}
	// Color image	
	else if (output.channels() == 3) {
		for (int counter = 0; counter < amount1; ++counter) {
            //r_salt
			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[0] = 0;
            //g_salt
			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[1] = 0;
            //b_salt
			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[2] = 0;
		}

		for (int counter = 0; counter < amount2; ++counter) {
            //r_pepper
			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[0] = 255;
            
			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[1] = 255;

			x = rng.uniform(0, output.rows);
			y = rng.uniform(0, output.cols);
			output.at<C>(x, y)[2] = 255;
		}
	}

	return output;
}

Mat Salt_pepper_noise_removal_Gray(const Mat input, int n, const char *opt) {
                                // (noise 있는 흑백 이미지, kernel 크기, boundary의 픽셀 처리 방법)
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int median;		// index of median value
    int tempx, tempy;

	// initialize median filter kernel
	Mat kernel;
	
	Mat output = Mat::zeros(row, col, input.type());

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
            kernel = Mat::zeros(kernel_size * kernel_size, 1, input.type());
            int count=0;
            int zero=0;
					
			if (!strcmp(opt, "zero-padding")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
                        count++;
								
                        if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
                            kernel.at<G>((x+n)*kernel_size+(y+n), 0) = input.at<G>(i+x, j+y);   //kernel에 각 pixel의 색상 데이터 저장
                        }
					}
				}
			}

			else if (!strcmp(opt, "mirroring")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
                        count++;
                        
                        //mirroring
                        if (i + x > row - 1)    tempx = i - x;
                        else if (i + x < 0)     tempx = -(i + x);
                        else                    tempx = i + x;
                    
                        if (j + y > col - 1)    tempy = j - y;
                        else if (j + y < 0)     tempy = -(j + y);
                        else                    tempy = j + y;
                        
                        kernel.at<G>((x+n)*kernel_size+(y+n), 0) = input.at<G>(tempx, tempy); //kernel에 각 pixel의 색상 데이터 저장
                        
                    }
                }
            }
			else if (!strcmp(opt, "adjustkernel")) {
                
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
                        if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
                            count++;    //adjustkernel은 이미지 범위 안에 들어가는 픽셀만 고려
                            kernel.at<G>((x+n)*kernel_size+(y+n), 0) = input.at<G>(i+x, j+y);   //kernel에 각 pixel의 색상 데이터 저장
                        }
                        else zero++;    //이미지 범위에서 벗어나는 pixel수를 센다.
					}
				}
			}

			// Sort the kernels in ascending order
			sort(kernel, kernel, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING); //오름차순 정렬
            median=zero+count/2;                                            //중앙값은 count한 pixel의 절반
            //adjustkernel인 경우 이미지 범위에서 벗어나 0인 데이터 값은 무시하기 위해 +zero
			output.at<G>(i, j) = kernel.at<G>(median, 0);                   //찾은 중앙값 output에 넣기
		}
	}

	return output;
}

Mat Salt_pepper_noise_removal_RGB(const Mat input, int n, const char *opt) {
                                // (noise 있는 컬러 이미지, kernel 크기, boundary의 픽셀 처리 방법)
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int median;		// index of median value
	int channel = input.channels();
    int tempx, tempy;
    

	// initialize ( (TypeX with 3 channel) - (TypeX with 1 channel) = 16 )
	// ex) CV_8UC3 - CV_8U = 16
	Mat kernel;
	
	Mat output = Mat::zeros(row, col, input.type());

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
            kernel = Mat::zeros(kernel_size * kernel_size, channel, input.type() - 16);
            int count=0;
            int zero=0;
            
			if (!strcmp(opt, "zero-padding")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
                        count++;
                                
                        if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
                            //kernel에 각 pixel의 색상 데이터 저장
                            kernel.at<G>((x+n)*kernel_size+(y+n), 0) = input.at<C>(i+x, j+y)[0];    // r
                            kernel.at<G>((x+n)*kernel_size+(y+n), 1) = input.at<C>(i+x, j+y)[1];    // g
                            kernel.at<G>((x+n)*kernel_size+(y+n), 2) = input.at<C>(i+x, j+y)[2];    // b
                        }
                       
					}
				}
			}

			else if (!strcmp(opt, "mirroring")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
                        count++;
                        
                        //mirroring
                        if (i + x > row - 1)    tempx = i - x;
                        else if (i + x < 0)     tempx = -(i + x);
                        else                    tempx = i + x;
                    
                        if (j + y > col - 1)    tempy = j - y;
                        else if (j + y < 0)     tempy = -(j + y);
                        else                    tempy = j + y;
                        
                        //kernel에 각 pixel의 색상 데이터 저장
                        kernel.at<G>((x+n)*kernel_size+(y+n), 0) = input.at<C>(tempx, tempy)[0];
                        kernel.at<G>((x+n)*kernel_size+(y+n), 1) = input.at<C>(tempx, tempy)[1];
                        kernel.at<G>((x+n)*kernel_size+(y+n), 2) = input.at<C>(tempx, tempy)[2];
                        
                    }
                }
            }

			else if (!strcmp(opt, "adjustkernel")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
                        if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
                            count++; //adjustkernel은 이미지 범위 안에 들어가는 픽셀만 고려
                           
                            //kernel에 각 pixel의 색상 데이터 저장
                            kernel.at<G>((x+n)*kernel_size+(y+n), 0) = input.at<C>(i+x, j+y)[0];
                            kernel.at<G>((x+n)*kernel_size+(y+n), 1) = input.at<C>(i+x, j+y)[1];
                            kernel.at<G>((x+n)*kernel_size+(y+n), 2) = input.at<C>(i+x, j+y)[2];
                        }
                        else zero++;    ////이미지 범위에서 벗어나는 pixel수를 센다
					}
				}
			}

			// Sort the kernels in ascending order
			sort(kernel, kernel, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);     //오름차순 정렬
            median=zero+count/2;                                                //중앙값은 count한 pixel의 절반
                                                                                //adjustkernel인 경우 이미지 범위에서 벗어나 0인 데이터 값은 무시하기 위해 +zero
            
            //찾은 중앙값 output에 넣기
			output.at<C>(i, j)[0] = kernel.at<G>(median, 0);
            output.at<C>(i, j)[1] = kernel.at<G>(median, 1);
            output.at<C>(i, j)[2] = kernel.at<G>(median, 2);
		}
	}

	return output;
}
