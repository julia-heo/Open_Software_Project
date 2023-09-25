실행하는 방법:
	Mac(M1)에서 OpenCV, Xcode 설치 후 사용

1. salt_and_pepper.cpp
코드의 목적: salt and pepper noise를 만들고, 이를 meanfilter를 사용해 제거한다.

함수 소개: Add_salt_pepper_Noise(const Mat input, float ps, float pp)
	매개 변수:
		input: input 이미지 행렬
		ps: density of salt noise (0~1) 
		pp: density of pepper noise (0~1)
	함수 목적:
		input 이미지에 ps 비율의 salt noise와 pp비율의 pepper noise를 만들어 반환

	Salt_pepper_noise_removal_Gray(const Mat input, int n, const char *opt)
	매개 변수:
		input: input 이미지 행렬(흑백)
		n: kernel의 크기 (2n+1)x(2n+1)
		opt: type of boundary processing. image boundary의 픽셀 처리 방법
			“zero-paddle” “mirroring” “adjustkernel” 중 하나
	함수 목적: 
		meanfilter를 이용해 salt and pepper noise를 제거한 이미지를 반환 (흑백)

	Mat Salt_pepper_noise_removal_RGB(const Mat input, int n, const char *opt);
	매개 변수:
		input: input 이미지 행렬(컬러)
		n: kernel의 크기 (2n+1)x(2n+1)
		opt: type of boundary processing. image boundary의 픽셀 처리 방법
			“zero-paddle” “mirroring” “adjustkernel” 중 하나
	함수 목적: 
		meanfilter를 이용해 salt and pepper noise를 제거한 이미지를 반환 (컬러)


2. Gaussian.cpp
코드의 목적: Gaussian noise를 만들고, 이를 Gaussian filter를 사용해 제거한다.
함수 소개: Add_Gaussian_noise(const Mat input, double mean, double sigma)
	매개변수:
		input: input 이미지 행렬
		mean: noise를 생성할 가우스 함수의 평균
		sigma: noise를 생성할 가우스 함수의 표준편차
	함수 목적:
		input 이미지에 평균을 mean, 표준편차를 sigma로 하는 가우스 함수에서 랜덤으로 가져온 수들이 값에 더해져 Gaussian Noise를 가지게 된 output 이미지 행렬을 반환

	Gaussianfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, const char *opt)
	매개변수:
		input: input 이미지 행렬(흑백)
		n: kernel의 크기 (2n+1)x(2n+1)
		sigma_t: x축에 대한 표준편차
		sigma_s: y축에 대한 표준편차
		opt: type of boundary processing. image boundary의 픽셀 처리 방법
			“zero-paddle” “mirroring” “adjustkernel” 중 하나
	함수 목적: Gaussian filter를 이용해 Gaussian noise를 제거한 이미지를 반환 (흑백)

	Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, const char *opt)
	매개변수:
		input: input 이미지 행렬(흑백)
		n: kernel의 크기 (2n+1)x(2n+1)
		sigma_t: x축에 대한 표준편차
		sigma_s: y축에 대한 표준편차
		opt: type of boundary processing. image boundary의 픽셀 처리 방법
			“zero-paddle” “mirroring” “adjustkernel” 중 하나
	함수 목적: Gaussian filter를 이용해 Gaussian noise를 제거한 이미지를 반환 (컬러)

3. Bilateral.cpp
코드의 목적: Gaussian noise를 만들고, 이를 Bilateral filtering를 사용해 제거한다.
함수 소개: Add_Gaussian_noise(const Mat input, double mean, double sigma)
	매개변수:
		input: input 이미지 행렬
		mean: noise를 생성할 가우스 함수의 평균
		sigma: noise를 생성할 가우스 함수의 표준편차
	함수 목적:
		input 이미지에 평균을 mean, 표준편차를 sigma로 하는 가우스 함수에서 랜덤으로 가져온 수들이 값에 더해져 Gaussian Noise를 가지게 된 output 이미지 행렬을 반환
	
	Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt)
	매개변수:
		input: input 이미지 행렬(흑백)
		n: kernel의 크기 (2n+1)x(2n+1)
		sigma_t: x축에 대한 표준편차
		sigma_s: y축에 대한 표준편차
		sigma_r: 명도에 대한 표준편차
		opt: type of boundary processing. image boundary의 픽셀 처리 방법
			“zero-paddle” “mirroring” “adjustkernel” 중 하나
	함수 목적: Bilateral filter를 이용해 Gaussian noise를 제거한 이미지를 반환 (흑백)

	Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt);	매개변수:
		input: input 이미지 행렬(흑백)
		n: kernel의 크기 (2n+1)x(2n+1)
		sigma_t: x축에 대한 표준편차
		sigma_s: y축에 대한 표준편차
		sigma_r: 명도에 대한 표준편차
		opt: type of boundary processing. image boundary의 픽셀 처리 방법
			“zero-paddle” “mirroring” “adjustkernel” 중 하나
	함수 목적: Bilateral filter를 이용해 Gaussian noise를 제거한 이미지를 반환 (컬러)
