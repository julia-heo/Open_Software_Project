실행하는 방법:
	Mac(M1)에서 OpenCV, Xcode 설치 후 사용

1. SIFT.cpp
코드의 목적: SIFT descriptor를 이용해 특징점이 되는 keypoint를 찾고, Affine Transform을 수행한다.

함수소개:
   euclidDistance(Mat& vec1, Mat& vec2)
	vec1과 vec2 사이의 거리(둘이 얼마나 다른지)를 반환한다. 유사할수록 값이 작다.

   nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors)
	keypoints를 가지는 descriptors위의 점 중 vec와 가장 유사한, matching되는 점의 인덱스를 반환한다.

   SecondNearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors, int first)
	keypoints를 가지는 descriptors위의 점 중 vec와 두번째(first 다음으로)로 유사한 점의 인덱스를 반환한다.

   findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1, vector<KeyPoint>& keypoints2, 
		Mat& descriptors2, vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold)
	descriptor1의 keypoints1에 matching되는 descriptor2의 keypoints2를 찾아 srcPoints(descriptor 1의 점)와 dstPoints(descriptor 2의 점)에 저장한다. 
	crossCheck: cross check여부, ratio_threshold: threshold ratio 사용 여부

   cal_affine:
	매개변수: int ptl_x[], int ptl_y[], int ptr_x[], int ptr_y[], int number_of_points
		ptl_x[]: corresponding pixels의 왼쪽 이미지에서의 x좌표
		ptl_y[]: corresponding pixels의 왼쪽 이미지에서의 y좌표
		ptr_x[]: corresponding pixels의 오른쪽 이미지에서의 x좌표
		ptr_y[]: corresponding pixels의 오른쪽 이미지에서의 y좌표
		number_of_points: corresponding pixels의 개수
	함수 목적: ptl_x, ptl_y와 계산해 ptr_x, ptr_y를 구할 수 있는 Matrix(A_12, A_21) 반환
   
   AffineTransform(Mat input1, Mat input2, vector<Point2f>& srcPoints, vector<Point2f>& dstPoints)
	input1과 input2의 matching된 keypoints의 좌표 담은 scrPoints와 dstPoints를 이용해 	Affine Transform을 수행한다. input2를 input1에 stitching한 결과를 반환한다

   SIFTfunc(Mat input1, Mat input2,vector<KeyPoint>& keypoints1, Mat& descriptors1, vector<KeyPoint>& keypoints2 , Mat& descriptors2)
	input1의 keypoint 정보를 담은 keypoints1과 descriptors1과 input2의 keypoint 정보를 담은 keypoints2과 descriptors2를 받아 findPairs() 함수를 이용해 featurematching을 진행하고, AffineTransform를 이용해 AffineTransform을 수행한다



2. SIFT_RANSAC.cpp
코드의 목적: SIFT descriptor를 이용해 특징점이 되는 keypoint를 찾고, Affine Transform을 수행한다. 이때, RANSAC을 적용한다.
특징: 1번 코드와 유사하지만, RANSAC을 적용하기에, A21과 A12를 구하는 과정에서 outlier가 제거된다.

3. Hough.cpp
코드의 목적: canny 함수를 이용해 egde를 구하고, HoughLines함수와 HoughLinesP함수를 이용해 edge를 선으로 그려낸다.
	HoughLinesP함수는 선의 시작점과 끝점이 있다는 특징이 있다.


