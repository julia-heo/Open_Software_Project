실행하는 방법:
	Mac(M1)에서 OpenCV, Xcode 설치 후 사용

1. LoG.cpp
코드의 목적: Laplacian of Gaussain을 적용하여 이미지의 Edge를 찾아낸다

함수소개:
   Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s);
	매개변수:
		input: input 이미지 행렬
		n: kernel의 크기 (2n+1)x(2n+1)
		sigma_t: x축에 대한 표준편차
		sigma_s: y축에 대한 표준편차
	함수 목적: Gaussian filter를 이용해 Gaussian noise를 제거한 이미지를 반환

   Laplacianfilter(const Mat input);	
	매개변수:
		input: input 이미지 행렬
	함수 목적: 
		input 이미지를 laplacianfiltering하여 결과 이미지을 리턴한다.

   get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize);
	매개변수:
		input: input 이미지 행렬
		n: kernel의 크기 (2n+1)x(2n+1)
		sigma_t: x축에 대한 표준편차
		sigma_s: y축에 대한 표준편차
		normalize: normalize 여부
	함수 목적: Gaussian Filter에 사용되는 Kernel을 계산하여 반환한다

   get_Laplacian_Kernel (int n, double sigma_t, double sigma_s, bool normalize);
	매개변수:
		input: input 이미지 행렬
		n: kernel의 크기 (2n+1)x(2n+1)
		sigma_t: x축에 대한 표준편차
		sigma_s: y축에 대한 표준편차
		normalize: normalize 여부
	함수 목적: Laplacian Filter에 사용되는 Kernel을 계산하여 반환한다
		
   Mirroring(const Mat input, int n);
	  매개변수:
		input: input 이미지 행렬
		n: kernel의 크기 (2n+1)x(2n+1)
	  함수 목적: input 이미지에 (2n+1)x(2n+1) kernel을 적용하기 위해 상하좌우로 2n만큼의 pixel에 mirroring방식으로 값을 채워준다.


2. LoG_RGB.cpp
코드의 목적: Laplacian of Gaussain을 적용하여 이미지의 Edge를 찾아낸다(컬러)
	1번 코드와 함수명, 수행하는 일은 같지만, 컬러이미지 처리를 위한 코드로 각 RGB에 대해 따로따로 연산이 진행된다.


3. Canny.cpp
코드의 목적: Opencv가 제공하는 함수를 이용해 Canny Edge Detector를 구현해본다.
함수 소개: Canny(InputArray, OutputArray, threshold1, threshold2)
	매개변수:
		InputArray: input 이미지 행렬
		OutputArray: 결과 반환할 행렬
		threshold1, threshold2: 기준점
	함수 목적: InputArray를 Image Filtering(Low-pass & High-pass filters)한 뒤, Non-	maximum suppression & Doouble thresholding 보정을 하여 OutputArray에 반환한다


4. Harris_corner.cpp
코드의 목적: 입력 이미지에 대해 Harris Corner Detector를 이용해 corner을 찾아낸다
주요 함수 소개: 
   NonMaximum_Suppression(const Mat input, Mat corner_mat, int radius)
		input: 입력 이미지 행렬
		corner_mat: corner이면 1, 아니면 0이 저장된 행렬
		radius: Non-maximum suppression에서 고려할 이웃들의 범위 반지름
	   함수 목적: corner pixel들에 접근해 주변 pixel 값 중 자신이 가장 클 때만 corner로 판단

   cornerHarris(img, output, blockSize, ksize, k, borderType)
		img: 입력 이미지
		output: 반환할 결과 이미지
		blockSize: corner detection에서 고려할 이웃 픽셀 크기
		ksize: (미분을 위한) 소벨 연산자를 위한 커널 크기
		k: 해리스 코너 검출 상수
		borderType: 가장자리 픽셀 확장 방식
	   함수목적: output에 해리스 코너 값 반환. 흑백이미지에서, 하얀 점이 코너이다.

         cornerSubPix(image, corners, winSize, zeroZone, criteria)
	image: input이미지
	corners: 코너점
	iwnSize: Search Window의 절반 사이즈
	zeroZone: Half of the size of the dead region in the middle of the search zone over which the summation in the formula below is not done
	criteria: 종료 시점
   함수목적: image의 컴출된 코너점의 조금 더 정확한 위치를 추출
