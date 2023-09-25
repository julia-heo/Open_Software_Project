실행하는 방법:
	Mac(M1)에서 OpenCV, Xcode 설치 후 사용

1. SIFT.cpp
코드의 목적: SIFT descriptor를 이용해 특징점이 되는 keypoint를 찾고, feature matching을 수행한다

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



2. feature_homography.cpp
코드의 목적: img1이 포함된 img2에 대해, matching되는 keypoints를 찾고, object의 테두리를 알아낸다