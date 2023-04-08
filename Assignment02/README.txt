실행하는 방법:
	Mac(M1)에서 OpenCV, Xcode 설치 후 사용

1. hist_func.h
코드의 목적: 이번 프로젝트에서 공통적으로 사용되는 헤더파일, 변수, 함수들을 정의한다.
함수 소개:
	cal_PDF (Mat &input)
		input: 입력 이미지 행렬의 주소값
	   함수 목적: input 이미지의 PDF를 리턴한다.

	cal_PDF_RGB (Mat &input)
		input: 입력 이미지 행렬의 주소값. 이때 이미지는 컬러이다.
	   함수 목적: 컬러 이미지의 PDF를 리턴한다
	
	cal_CDF (Mat &input)
		input: 입력 이미지 행렬의 주소값
	   함수 목적: input 이미지의 CDF를 리턴한다.
	
	cal_CDF_RGB (Mat &input)
		input: 입력 이미지 행렬의 주소값. 이때 이미지는 컬러이다.
	   함수 목적: 컬러 이미지의 PDF를 리턴한다



2. PDF_CDF.cpp
코드의 목적: 흑백 이미지를 입력받아 PDF와 CDF를 계산하고 계산 결과를 출력하는 텍스트 파일을 생성한다.
	hist_func.h의 cal_PDF() 함수와 cal_CDF()함수를 호출한다

3. hist_stretching.cpp
코드의 목적: linear strectching function을 통해 histogram stretching한 결과 이미지를 새로운 창으로 띄우고,계산 결과를 출력하는 텍스트 파일을 생성한다.
함수 소개:
	inear_stretching(Mat &input, Mat &stretched, G *trans_func, G x1, G x2, G y1, G y2)
		input: 입력 이미지. 흑백이다
		stretched: histogram stretching의 결과를 저장할 Mat
		trans_func: mapping될 색 저장할 배열
		x1,x2: pixel몰린 구간
		y1,y2: pixel을 집중적으로 나눠주고싶은 구간
	   함수 목적:  input이미지를 histogram stretching해 stretched 와 trans_func에 결과값을 넣어준다

4. hist_eq.cpp
코드의 목적: Histogram equalization을 통해 흑백 이미지의 contrast를 증가시킨다. 결과 이미지를 새로운 창으로 띄우고, 계산 결과를 출력하는 텍스트 파일을 생성한다.
함수 소개: 
	hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF)
		input: 입력 이미지. 흑백이다
		equalized: Histogram equalization의 결과를 저장할 Mat
		trans_func: mapping될 색 저장할 배열
		CDF: 입력 이미지의 CDF
	   함수 목적: 입력받은 이미지에 대해 histogram equalization 수행

5. hist_eq_RGB.cpp
코드의 목적: Histogram equalization을 통해 컬러 이미지의 contrast를 증가시킨다. 결과 이미지를 새로운 창으로 띄우고, 계산 결과를 출력하는 텍스트 파일을 생성한다.
함수 소개:
	hist_eq_Color(Mat &input, Mat &equalized, G(*trans_func)[3], float **CDF) 
		input: 입력 이미지. 흑백이다
		equalized: Histogram equalization의 결과를 저장할 Mat
		trans_func: mapping될 색 저장할 배열
		CDF: 입력 이미지의 CDF
	   함수 목적: RGB(컬러) 이미지 데이터에 대해 Histogram Equalization을 진행한다

6. hist_eq_YUV.cpp
코드의 목적: 컬러 이미지의 Histogram Equalization을 진행한다. RGB의 이미지 데이터를 YUV로 변환한 뒤, Y 값에 대해서만 Histogram Equalization해준다. 
함수 소개:
	hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF)
		input: Histogram Equalization할 데이터. 여기선 Y채널의 값만이 전달될 것이다.
		equalized: Histogram equalization의 결과를 저장할 Mat
		trans_func: Y가 mapping될 값 저장할 배열
		CDF: Histogram Equalization할 데이터의 CDF
	   함수 목적: 채널 개수가 1개인 데이터에 대해 Histogram Equalization을 진행한다.

7. hist_hm_YUV.cpp
코드의 목적: 컬러 이미지의 Histogram Matching을 진행한다. RGB의 이미지 데이터를 YUV로 변환한 뒤, Y 값에 대해서만 Histogram Matching해준다. 
함수 소개:
	hist_ma(Mat &input, Mat &reference, Mat &equalized, G *trans_func, float *CDF,float *CDF_reference)
		input: HM할 데이터. 여기선 Y채널의 값만이 전달될 것이다.
		reference: HM의 reference 이미지
		equalized: HM의 결과를 저장할 Mat
		trans_func: Y가 mapping될 값 저장할 배열
		CDF: HM할 데이터의 CDF
		CDF_reference: 레퍼런스 이미지의 CDF
	   함수 목적: input이미지를 reference이미지의 히스토그램에 가까워지도록 Histogram Matching을 수행한다




