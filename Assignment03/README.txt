실행하는 방법:
	Mac(M1)에서 OpenCV, Xcode 설치 후 사용

1-1. MeanFilterGray.cpp
코드의 목적: 흑백 이미지에 대해 simplest low-pass filter인 Uniform Mean filter를 구현한다.
	kernel 크기와 image boundary의 픽셀 처리 방법에 따라 다른 결과 이미지를 출력한다

함수 소개: meanfilter(const Mat input, int n, const char* opt)
	매개변수:
		input: input 흑백 이미지 행렬
		n: kernel의 크기 (2n+1)x(2n+1)
		opt: type of boundary processing. image boundary의 픽셀 처리 방법
			“zero-paddle” “mirroring” “adjustkernel” 중 하나
	함수 목적:
		input을 opt 적용해 mean filtering한 결과를 반환(Mat)


1-2. MeanFilterRGB.cpp
코드의 목적: 컬러 이미지에 대해simplest low-pass filter인 Uniform Mean filter를 구현한다.
	kernel 크기와 image boundary의 픽셀 처리 방법에 따라 다른 결과 이미지를 출력한다.
함수 소개: meanfilter(const Mat input, int n, const char* opt)
	매개변수:
		input: input 이미지 행렬. 컬러이미지
		n: kernel의 크기 (2n+1)x(2n+1)
		opt: type of boundary processing. image boundary의 픽셀 처리 방법
			“zero-paddle” “mirroring” “adjustkernel” 중 하나
	함수 목적:
		input을 opt 적용해 mean filtering한 결과를 반환(Mat)


2-1. GaussianGray.cpp
코드의 목적: 흑백 이미지에 대해 Gaussian filter를 구현한다.
	kernel 크기와 x축에 대한 표준편차, y축에 대한 표준편차, image boundary의 픽셀 처리 방법에 따라 다른 결과 이미지를 출력한다.
함수 소개: gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt)
	  매개변수:
		input: input 흑백 이미지 행렬
		n: kernel의 크기 (2n+1)x(2n+1)
		sigmaT, sigmaS: σt σs 표준편차
		opt: type of boundary processing. image boundary의 픽셀 처리 방법
			“zero-paddle” “mirroring” “adjustkernel” 중 하나
	함수 목적:
		input을 opt 적용해 Gaussian filtering한 결과를 반환(Mat)
 

2-2. GaussianRGB.cpp
코드의 목적: 컬러 이미지에 대해 Gaussian filter를 구현한다.
함수 소개: gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt)
	  매개변수:
		input: input 흑백 이미지 행렬
		n: kernel의 크기 (2n+1)x(2n+1)
		sigmaT, sigmaS: σt σs 표준편차
		opt: type of boundary processing. image boundary의 픽셀 처리 방법
			“zero-paddle” “mirroring” “adjustkernel” 중 하나
	함수 목적:
		input을 opt 적용해 Gaussian filtering한 결과를 반환(Mat)


3-1. SobelGray.cpp
코드의 목적: 흑백 이미지에 대해 Sobel filter를 구현한다. boundary processing은 mirroring을 사용한다.
함수 소개: sobelfilter ( const Mat input )
	매개변수:
		input: 입력 이미지 행렬
	함수 목적:
		input 이미지를 sobel filtering한다.


3-2. SobelRGB.cpp
코드의 목적: 컬러 이미지에 대해 Sobel filter를 구현한다. boundary processing은 mirroring을 사용한다.
함수 소개: : sobelfilter ( const Mat input )
	매개변수:
		input: 입력 이미지 행렬. 컬러이기 때문에 3개의 채널을 가지고 있다.
	함수 목적:
		input 이미지를 sobel filtering한다.


4-1. LaplacianGray.cpp
코드의 목적: 흑백 이미지에 대해 Laplacian filter를 구현한다. boundary processing은 mirroring을 사용한다.
함수 소개: laplacianfilter ( const Mat input )
	매개변수:
		input: 입력 이미지 행렬. 흑백
	함수 목적:
		input 이미지를 laplacianfiltering하여 결과 이미지(흑백)을 리턴한다.


4-2. LaplacianRGB.cpp
코드의 목적: 컬러 이미지에 대해 Laplacian filter를 구현한다. boundary processing은 mirroring을 사용한다.
함수 소개: laplacianfilter ( const Mat input )
	매개변수:
		input: 입력 이미지 행렬. 흑백
	함수 목적:
		input 이미지를 Laplacian filtering하여 결과 이미지(흑백)을 리턴한다.


5-1. Gaussian_sep_Gray.cpp
코드의 목적: 2번의 Gaussian Filtering에서, 연산량을 줄이기 위해 를 두개의 filter로 나눠 계산한다. kernel 크기와 x축에 대한 표준편차, y축에 대한 표준편차, image boundary의 픽셀 처리 방법에 따라 다른 결과 이미지를 출력한다.
함수 소개: Gaussianfilter_sep (const Mat input, int n, float sigmaT, float sigmaS, const char* opt)
	  매개변수:
		input: input 흑백 이미지 행렬
		n: kernel의 크기 (2n+1)x(2n+1)
		sigmaT, sigmaS: σt σs 표준편차
		opt: type of boundary processing. image boundary의 픽셀 처리 방법
			“zero-paddle” “mirroring” “adjustkernel” 중 하나
	함수 목적:
		input을 opt 적용해 Gaussian filtering한 결과를 반환(Mat)


5-2. Gaussian_sep_RGB.cpp
코드의 목적: 2번의 Gaussian Filtering에서, 연산량을 줄이기 위해 를 두개의 filter로 나눠 계산한다. kernel 크기와 x축에 대한 표준편차, y축에 대한 표준편차, image boundary의 픽셀 처리 방법에 따라 다른 결과 이미지를 출력한다. 이미지는 컬러 이미지이다. 
함수 소개: Gaussianfilter_sep (const Mat input, int n, float sigmaT, float sigmaS, const char* opt)
	  매개변수:
		input: input 컬러 이미지 행렬
		n: kernel의 크기 (2n+1)x(2n+1)
		sigmaT, sigmaS: σt σs 표준편차
		opt: type of boundary processing. image boundary의 픽셀 처리 방법
			“zero-paddle” “mirroring” “adjustkernel” 중 하나
	함수 목적:
		input을 opt 적용해 Gaussian filtering한 결과를 반환(Mat)


6-1. UnsharpMaskGray.cpp
코드의 목적: 흑백 이미지에 대해 Unsharp Masking를 구현한다.
	kernel 크기와 표준편차, image boundary의 픽셀 처리 방법에 따라 Gaussian filtering을 진행하고, 이를 low pass filtering으로 하는 Unsharp Masking을 진행한다.
함수 소개: UnsharpMaskfilter (const Mat input, int n, float sigmaT, float sigmaS, const char* opt, float k);
	매개변수:
		input: 흑백 input이미지
		n: kernel의 크기 (2n+1)x(2n+1)
		sigmaT: x축에 대한 표준편차
 		sigmaS: y축에 대한 표준편차
		opt: type of boundary processing. image boundary의 픽셀 처리 방법
			“zero-paddle” “mirroring” “adjustkernel” 중 하나
		k: losspass filter 반영 비율
	함수 목적: input을 opt 적용해 Unsharp Masking한 결과를 반환(Mat)

	gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt)
	  매개변수: input: input 흑백 이미지 행렬
		n: kernel의 크기 (2n+1)x(2n+1)
		sigmaT, sigmaS: σt σs 표준편차
		opt: type of boundary processing. image boundary의 픽셀 처리 방법
			“zero-paddle” “mirroring” “adjustkernel” 중 하나
	함수 목적: input을 opt 적용해 Gaussian filtering한 결과를 반환(Mat)


6-2. UnsharpMaskRGB.cpp
코드의 목적: 흑백 이미지에 대해 Unsharp Masking를 구현한다.
	kernel 크기와 표준편차, image boundary의 픽셀 처리 방법에 따라 Gaussian filtering을 	진행하고, 이를 low-pass filtering으로 하는 Unsharp Masking을 진행한다.
함수 소개: UnsharpMaskfilter (const Mat input, int n, float sigmaT, float sigmaS, const char* opt, float k);
	매개변수:
		input: 컬러 input이미지
		n: kernel의 크기 (2n+1)x(2n+1)
		sigmaT: x축에 대한 표준편차
 		sigmaS: y축에 대한 표준편차
		opt: type of boundary processing. image boundary의 픽셀 처리 방법
			“zero-paddle” “mirroring” “adjustkernel” 중 하나
		k: losspass filter 반영 비율
	함수 목적:
		input을 opt 적용해 Unsharp Masking한 결과를 반환(Mat)

	gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt)
	  매개변수:
		input: input 컬러 이미지 행렬
		n: kernel의 크기 (2n+1)x(2n+1)
		sigmaT, sigmaS: σt σs 표준편차
		opt: type of boundary processing. image boundary의 픽셀 처리 방법
			“zero-paddle” “mirroring” “adjustkernel” 중 하나
	함수 목적:
		input을 opt 적용해 Gaussian filtering한 결과를 반환(Mat)

