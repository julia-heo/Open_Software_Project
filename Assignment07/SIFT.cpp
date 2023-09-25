#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#define RATIO_THR 0.4

using namespace std;
using namespace cv;

double euclidDistance(Mat& vec1, Mat& vec2);
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
int SecondNearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors, int first);
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
            vector<KeyPoint>& keypoints2, Mat& descriptors2,
            vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold);
int main() {
    //입력 이미지
    Mat input1 = imread("input1.jpg", CV_LOAD_IMAGE_COLOR);
    Mat input2 = imread("input2.jpg", CV_LOAD_IMAGE_COLOR);
    Mat input1_gray, input2_gray;

    if (!input1.data || !input2.data)
    {
        std::cout << "Could not open" << std::endl;
        return -1;
    }

    //resize(input1, input1, Size(input1.cols / 2, input1.rows / 2));
    //resize(input2, input2, Size(input2.cols / 2, input2.rows / 2));

    //흑백 전환
    cvtColor(input1, input1_gray, CV_RGB2GRAY);
    cvtColor(input2, input2_gray, CV_RGB2GRAY);
    
    //-----------------------------------------------
    
    FeatureDetector* detector = new SiftFeatureDetector(
        0,        // nFeatures
        4,        // nOctaveLayers
        0.04,    // contrastThreshold
        10,        // edgeThreshold
        1.6        // sigma
    );

    DescriptorExtractor* extractor = new SiftDescriptorExtractor();

    //2 to 1을 위한 변수 설정
    // Create a image for displaying mathing keypoints
    Size size = input2.size();
    Size sz = Size(size.width + input1_gray.size().width, max(size.height, input1_gray.size().height));
    Mat matchingImage = Mat::zeros(sz, CV_8UC3);

    input1.copyTo(matchingImage(Rect(size.width, 0, input1_gray.size().width, input1_gray.size().height)));
    input2.copyTo(matchingImage(Rect(0, 0, size.width, size.height)));

    
    //1 to 2를 위한 변수 설정
    vector<Point2f> srcPoints2;
    vector<Point2f> dstPoints2;
    Size size2 = input1.size();
    Size sz2 = Size(size2.width + input2_gray.size().width, max(size2.height, input2_gray.size().height));
    Mat matchingImage2 = Mat::zeros(sz2, CV_8UC3);
    
    //이미지 합치기
    input2.copyTo(matchingImage2(Rect(size2.width, 0, input2_gray.size().width, input2_gray.size().height)));
    input1.copyTo(matchingImage2(Rect(0, 0, size2.width, size2.height)));
    
    //-----------------------------------------------
    
    // Compute keypoints and descriptor from the source image in advance
    // input1 이미지의 keypoint 찾기
    vector<KeyPoint> keypoints1;
    Mat descriptors1;
    // Detect keypoints
    detector->detect(input1_gray, keypoints1);
    extractor->compute(input1_gray, keypoints1, descriptors1);
    printf("input1 : %d keypoints are found.\n", (int)keypoints1.size());

    // input2 이미지의 keypoint 찾기
    vector<KeyPoint> keypoints2;
    Mat descriptors2;
    // Detect keypoints
    detector->detect(input2_gray, keypoints2);
    extractor->compute(input2_gray, keypoints2, descriptors2);
    printf("input2 : %zd keypoints are found.\n", keypoints2.size());

    // input1의 keypoint에 aqua색으로 원을 그린다
    for (int i = 0; i < keypoints1.size(); i++) {
        KeyPoint kp = keypoints1[i];
        kp.pt.x += size.width;
        circle(matchingImage, kp.pt, cvRound(kp.size*0.25), Scalar(255, 255, 0), 1, 8, 0);
        circle(matchingImage2, kp.pt, cvRound(kp.size*0.25), Scalar(255, 255, 0), 1, 8, 0);
    }
    // input2의 keypoint에 aqua색으로 원을 그린다
    for (int i = 0; i < keypoints2.size(); i++) {
        KeyPoint kp = keypoints2[i];
        circle(matchingImage, kp.pt, cvRound(kp.size*0.25),
               Scalar(255, 255, 0), 1, 8, 0);
        circle(matchingImage2, kp.pt, cvRound(kp.size*0.25),
               Scalar(255, 255, 0), 1, 8, 0);
    }
    
    //---------------------------------------------

    // Find nearest neighbor pairs 2 to 1
    vector<Point2f> srcPoints;
    vector<Point2f> dstPoints;
    bool crossCheck = true;
    bool ratio_threshold = true;
    
    findPairs(keypoints2, descriptors2, keypoints1, descriptors1, srcPoints, dstPoints, crossCheck, ratio_threshold);   //keypoint를 matching한다
    printf("%zd keypoints are matched.2to1\n", srcPoints.size());

    // Draw line between nearest neighbor pairs
    // matching된 점들을 빨간 선으로 연결한다
    for (int i = 0; i < (int)srcPoints.size(); ++i) {
        Point2f pt1 = srcPoints[i];
        Point2f pt2 = dstPoints[i];
        Point2f from = pt1;
        Point2f to = Point(size.width + pt2.x, pt2.y);
        line(matchingImage, from, to, Scalar(0, 0, 255));
    }

    // Display mathing image
    namedWindow("Matching 2to1");
    imshow("Matching 2to1", matchingImage);
    
    //---------------------------------------------

    // Find nearest neighbor pairs 1 to 2
    findPairs(keypoints1, descriptors1, keypoints2, descriptors2, srcPoints2, dstPoints2, crossCheck, ratio_threshold);         //keypoint를 matching한다
    printf("%zd keypoints are matched.(1to2)\n", srcPoints2.size());

    // Draw line between nearest neighbor pairs
    for (int i = 0; i < (int)srcPoints.size(); ++i) {
        Point2f pt1 = srcPoints2[i];
        Point2f pt2 = dstPoints2[i];
        Point2f from = pt1;
        Point2f to = Point(size2.width + pt2.x, pt2.y);
        line(matchingImage2, from, to, Scalar(0, 0, 255));
    }
    // Display mathing image
    namedWindow("Matching 1to2");
    imshow("Matching 1to2", matchingImage2);

    waitKey(0);

    return 0;
}

//Calculate euclid distance
double euclidDistance(Mat& vec1, Mat& vec2) {
    // vec1과 vec2 사이의 거리를 반환
    double sum = 0.0;
    int dim = vec1.cols;
    for (int i = 0; i < dim; i++) {
        sum += (vec1.at<uchar>(0, i) - vec2.at<uchar>(0, i)) * (vec1.at<uchar>(0, i) - vec2.at<uchar>(0, i));
    }

    return sqrt(sum);
}

// Find the index of nearest neighbor point from keypoints.
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
    // keypoints가 담긴 벡터
    int neighbor = -1;
    double minDist = 1e6;

    for (int i = 0; i < descriptors.rows; i++) {
        Mat v = descriptors.row(i);        // each row of descriptor
        double dist = euclidDistance(vec, v);
        if (dist < minDist) {
            // 거리가 가장 가까운(feature이 가장 비슷한) 점을 matching되는 점으로 한다
            minDist = dist;
            neighbor = i;
        }
    }

    return neighbor;
}
//---------------------------------------------------
int SecondNearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors,int first) {
    //
    int neighbor = -1;
    double minDist = 1e6;

    for (int i = 0; i < descriptors.rows; i++) {
        Mat v = descriptors.row(i);        // each row of descriptor
        double dist = euclidDistance(vec, v);
        if (dist < minDist) {
            if(i==first)continue;   //가장 작은 keypoint면 continue
            minDist = dist;
            neighbor = i;
        }
    }

    return neighbor;
}
//---------------------------------------------------


//Find pairs of points with the smallest distace between them
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
    vector<KeyPoint>& keypoints2, Mat& descriptors2,
    vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold) {
    //keypoints1을 keypoints2에 매칭하고, 값을 srcPoints와 dstPoints에 반환
    for (int i = 0; i < descriptors1.rows; i++) {   //descriptors1의 keypoints들에 대해
        KeyPoint pt1 = keypoints1[i];
        Mat desc1 = descriptors1.row(i);

        int nn = nearestNeighbor(desc1, keypoints2, descriptors2);  // nearest neighbors

        // Refine matching points using ratio_based thresholding
        if (ratio_threshold) {  //ratio_threshold가 true인 경우 수행
            int nn2=SecondNearestNeighbor(desc1, keypoints2, descriptors2,nn); //두번째로 가까운 keypoint를 반환

            Mat v1=descriptors2.row(nn);
            double dist1 = euclidDistance(desc1, v1);

            Mat v2=descriptors2.row(nn2);
            double dist2 = euclidDistance(desc1, v2);

            if ((dist1/dist2) > 0.65) continue; //가장 가까운 keypoint와 두번째 가까운 keypoint와의 거리를 threshold로 가진다
        }

        // Refine matching points using cross-checking
        if (crossCheck) {   //crossCheck가 true인 경우 수행
            Mat desc2 = descriptors2.row(nn);
            int nn2= nearestNeighbor(desc2, keypoints1, descriptors1); // 매칭된 keypoint에 대해 반대로 nn 수행
            if(nn2!=i) continue;    //서로가 서로의 가장 가까운 feature일 때만 matching

        }

        KeyPoint pt2 = keypoints2[nn];
        srcPoints.push_back(pt1.pt);
        dstPoints.push_back(pt2.pt);
    }
}

