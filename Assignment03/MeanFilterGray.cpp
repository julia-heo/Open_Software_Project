#include <iostream>             //헤더파일 선언
#include <opencv2/opencv.hpp>

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

Mat meanfilter(const Mat input, int n, const char* opt);    //meanfilter 함수

int main() {
	
	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);    //"lena.jpg" 이미지를 컬러로 불러와서 input 변수에 대입
	Mat input_gray;
	Mat output;

	cvtColor(input, input_gray, CV_RGB2GRAY); // Converting image to gray
	//input을 흑백으로 전환해 input_gray에 저장

	if (!input.data)
	{
        std::cout << "Could not open" << std::endl;         //파일이 열리지 않았다면 "Could not open"출력 후 종료
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);       //흑백의 input 이미지 새로운 창으로 띄우기
	imshow("Grayscale", input_gray);
	output = meanfilter(input_gray,15, "zero-paddle"); //Boundary process: zero-paddle, mirroring, adjustkernel
        //output 행렬에 meanfilter의 결과를 저장
        //매개변수: input이미지(mean filtering할 이미지), N (filter kernel (2N+1)x(2N+1)), type of boundary processing

	namedWindow("Mean Filter", WINDOW_AUTOSIZE);    //결과 이미지 새로운 창으로 띄우기
	imshow("Mean Filter", output);


	waitKey(0); //대기함수

	return 0;
}


Mat meanfilter(const Mat input, int n, const char* opt) {

	Mat kernel; //filtering kernel
	
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
    
    // Initialiazing Kernel Matrix
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
        //ones(a, b, CV_...) :  a행 b열의 CV_... 타입 모든 원소가 1인 행렬을 반환,(kernel_size * kernel_size)로 나눔
        //전체에 같은 가중치를 두고, 전체 필터 크기로 나눠 평균을 구한다
        //ex n=3 : filter kernel = [ 1/9, 1/9, 1/9; 1/9, 1/9, 1/9; 1/9, 1/9, 1/9 ]
    
	float kernelvalue=kernel.at<float>(0, 0);  // To simplify, as the filter is uniform. All elements of the kernel value are same.
	// mean filter라 kernel안의 모든 픽셀에 대한 가중치가 같으므로 kernelvalue에 가중치 값을 저장하여 사용한다. 1/(n^2)
    
	Mat output = Mat::zeros(row, col, input.type());
	//zeros(a, b, CV_...) : a행 b열의 CV_... 타입 영행렬을 반환
	
	for (int i = 0; i < row; i++) { //for each pixel in the output
		for (int j = 0; j < col; j++) {
			
			if (!strcmp(opt, "zero-paddle")) {  //Zero padding: 이미지의 boundary를 0으로 채운다
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
                            //if the pixel is not a border pixel 이미지 범위 안의 픽셀이라면
							sum1 += kernelvalue*(float)(input.at<G>(i + a, j + b)); //해당 픽셀의 이미지 데이터에 filtering의 가중치를 곱한 값을 sum1에 더함
						} //이미지에서 벗어난 boundary는 더해지지 않음. 즉 0으로 반영됨
					}
				}
				output.at<G>(i, j) = (G)sum1;   //필터링해 얻은 값을 i,j픽셀에 넣어준다
			}
			
			else if (!strcmp(opt, "mirroring")) {   //Mirroring: 이미지의 경계를 벗어난 픽셀은 반사된 값으로 boundary를 채운다
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {
                        
                        //mirroring for the border pixels
						if (i + a > row - 1) {      //kernel이 이미지의 오른쪽으로 삐져나간 경우
							tempa = i - a;          //i를 기준으로 mirroring. i-a픽셀의 값을 가져올 것이다.
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
						sum1 += kernelvalue*(float)(input.at<G>(tempa, tempb)); //kernel 안의 모든 픽셀에 대해 가중치를 곱해 sum1에 더한다
					}
				}
				output.at<G>(i, j) = (G)sum1;   //(i,j) 픽셀을 filtering한 값을 output의 (i,j)에 저장
			}
			
			else if (!strcmp(opt, "adjustkernel")) { //adjusting: 이미지 범위에 벗어나는 pixel은 아예 반영하지 않는다. kernel이 아미지의 범위를 벗어나면 kernel을 자른다
				float sum1 = 0.0;
				float sum2 = 0.0;   //kernel에 해당하는 픽셀의 비율을 더하는 변수
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //이미지 범위에 벗어나지 않는다면
							sum1 += kernelvalue*(float)(input.at<G>(i + a, j + b));     //해당 픽셀의 이미지 데이터에 filtering의 가중치를 곱한 값을 sum1에 더함
							sum2 += kernelvalue;    //이미지 범위 안에 들어가는 pixel의 가중치만을 더한다.
						}
					}
				}
				output.at<G>(i, j) = (G)(sum1/sum2);
                    //계산에 반영된 pixe들의 가중치 합으로 나눠준다. 이미지 안에 포함되는 kernel안의 pixel들만의 평균울 구할 수 있다.
                    //모든 kernel이 이미지 안에 있는 경우 sum2는 1
			}
		}
	}
	return output;
}
