실행하는 방법:
	Mac(M1)에서 OpenCV, Xcode 설치 후 사용

1. adaptivethreshold.cpp
코드의 목적: Moving Average를 이용한 Adaptive Thresholding을 적용한다.

함수 소개: adaptive_thres(const Mat input, int n, float b)
	매개변수:
		input: input 이미지 행렬
		n: kernel의 크기 (2n+1)x(2n+1)
		b: weight
	함수 목적: kernel의 mean에 b의 가중치를 준 것을 Threshold로 하여 input 이미지를 Adaptive Thresholding하여 반환

2. kmeans.cpp
코드의 목적: 흑백 이미지에 대한 K-means Clustering을 진행한다. intensity만을 고려한 Clustering과 position까지 고려한 Clustering에 대해 각각 결과를 도출한다.
함수 소개: Kmeans(const Mat input,int clusterCount, int attempts)
		매개변수:
			input: input 이미지 행렬
			clusterCount: 군집화할 개수(k)
			attempts: 실행되는 알고리즘의 실행 횟수
		함수 목적:
			input 이미지의 데이터를 clustering하고, 결과값에 맞게 output이미지를 만들어 반환한다.

	KmeansPosition(const Mat input,int clusterCount, int attempts ,float sigmaX, float sigmaY)
		매개변수:
			input: input 이미지 행렬
			clusterCount: 군집화할 개수(k)
			attempts: 실행되는 알고리즘의 실행 횟수
			sigma: intensity와 position의 다른 ratio를 맞춰 주기 위한 상수
		함수 목적:
			input 이미지의 데이터를 intensity와 position에 따라 clustering하고, 결과값에 맞게 output이미지를 만들어 반환한다.

     	cv::kmeans( InputArray data, int K, InputOutputArray bestLabels, TermCriteria criteria, int attempts, int flags, OutputArray centers = noArray())
		매개 변수:
			data : input 이미지
			K : 군집화할 개수
			(반환값) bestLabels : 라벨에 대한 배열. 가장 가까운 center의 인덱스 
			criteria : 반복을 종료할 조건 (type, max_iter최대반복횟수, epsilon정확도)
			attempts : 다른 초기 라벨링을 사용하면서 실행되는 알고리즘의 실행 횟수를 지정하는 플래그
			flags : 초기값을 잡을 중심에 대한 플래그로써 cv2.KMEANS_PP_CENTERS와 cv2.KMEANS_RANDOM_CENTERS 중 하나
			(반환값) centers : 클러스터의 중심이 저장된 배열
		함수 목적: samples를 k개의 group으로 cluster하여 해당하는 그룹의 center인덱스와 center의 색을 반환한다.

3. kmeansRGB.cpp
코드의 목적: 컬러 이미지에 대한 K-means Clustering을 진행한다. intensity만을 고려한 Clustering과 position까지 고려한 Clustering에 대해 각각 결과를 도출한다.
함수 소개: Kmeans(const Mat input,int clusterCount, int attempts)
		매개변수:
			input: input 이미지 행렬
			clusterCount: 군집화할 개수(k)
			attempts: 실행되는 알고리즘의 실행 횟수
		함수 목적:
			input 이미지의 데이터를 clustering하고, 결과값에 맞게 output이미지를 만들어 반환한다.

	KmeansPosition(const Mat input,int clusterCount, int attempts ,float sigmaX, float sigmaY)
		매개변수:
			input: input 이미지 행렬
			clusterCount: 군집화할 개수(k)
			attempts: 실행되는 알고리즘의 실행 횟수
			sigma: intensity와 position의 다른 ratio를 맞춰 주기 위한 상수
		함수 목적:
			input 이미지의 데이터를 intensity와 position에 따라 clustering하고, 결과값에 맞게 output이미지를 만들어 반환한다.

     	cv::kmeans( InputArray data, int K, InputOutputArray bestLabels, TermCriteria criteria, int attempts, int flags, OutputArray centers = noArray())
		매개 변수:
			data : input 이미지
			K : 군집화할 개수
			(반환값) bestLabels : 라벨에 대한 배열. 가장 가까운 center의 인덱스 
			criteria : 반복을 종료할 조건 (type, max_iter최대반복횟수, epsilon정확도)
			attempts : 다른 초기 라벨링을 사용하면서 실행되는 알고리즘의 실행 횟수를 지정하는 플래그
			flags : 초기값을 잡을 중심에 대한 플래그로써 cv2.KMEANS_PP_CENTERS와 cv2.KMEANS_RANDOM_CENTERS 중 하나
			(반환값) centers : 클러스터의 중심이 저장된 배열
		함수 목적: samples를 k개의 group으로 cluster하여 해당하는 그룹의 center인덱스와 center의 색을 반환한다.
	