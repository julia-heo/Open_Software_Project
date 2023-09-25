#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#define RATIO_THR 0.4

using namespace std;
using namespace cv;


Mat SIFTfunc(Mat input1, Mat input2,vector<KeyPoint>& keypoints1, Mat& descriptors1,vector<KeyPoint>& keypoints2, Mat& descriptors2);
Mat AffineTransform(Mat input1, Mat input2, vector<Point2f>& srcPoints, vector<Point2f>& dstPoints);
double euclidDistance(Mat& vec1, Mat& vec2);
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
int SecondNearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors, int first);
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
            vector<KeyPoint>& keypoints2, Mat& descriptors2,
            vector<Point2f>& srcPoints, vector<Point2f>& dstPoints);
template <typename T>
Mat cal_affine(int ptl_x[], int ptl_y[], int ptr_x[], int ptr_y[], int number_of_points);
void blend_stitching(const Mat I1, const Mat I2, Mat &I_f, int diff_x, int diff_y, float alpha);
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

    //-----------------------------------------------
    
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
    
    //-----------------------------------------------
    
    Mat I_f=SIFTfunc(input1, input2, keypoints1,descriptors1,keypoints2,descriptors2);
    Mat I_f2=SIFTfunc(input2, input1, keypoints2,descriptors2,keypoints1,descriptors1);
    
    //"Left Image"창에 I1 이미지 띄우기
    namedWindow("input1");
    imshow("input1", input1);

    //"Right Image"창에 input2 이미지 띄우기
    namedWindow("input2");
    imshow("input2", input2);

    //"result"창에 I_f(결과물) 이미지 띄우기
    namedWindow("result(2to1)");
    imshow("result(2to1)", I_f);

    //"result"창에 I_f(결과물) 이미지 띄우기
    namedWindow("result2(1to2)");
    imshow("result2(1to2)", I_f2);
    
    
    
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


void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
    vector<KeyPoint>& keypoints2, Mat& descriptors2,
    vector<Point2f>& srcPoints, vector<Point2f>& dstPoints) {
    for (int i = 0; i < descriptors1.rows; i++) {   //descriptors1의 keypoints들에 대해
        KeyPoint pt1 = keypoints1[i];
        Mat desc1 = descriptors1.row(i);

        int nn = nearestNeighbor(desc1, keypoints2, descriptors2);  // nearest neighbors

        //ratio_threshold
        int nn2=SecondNearestNeighbor(desc1, keypoints2, descriptors2,nn); //두번째로 가까운 keypoint를 반환

        Mat v1=descriptors2.row(nn);
        double dist1 = euclidDistance(desc1, v1);

        Mat v2=descriptors2.row(nn2);
        double dist2 = euclidDistance(desc1, v2);

        if ((dist1/dist2) > 0.65) continue; //가장 가까운 keypoint와 두번째 가까운 keypoint와의 거리를 threshold로 가진다
        
        // cross-checking
        Mat desc2 = descriptors2.row(nn);
        int nn_2= nearestNeighbor(desc2, keypoints1, descriptors1); // 매칭된 keypoint에 대해 반대로 nn 수행
        if(nn_2!=i) continue;    //서로가 서로의 가장 가까운 feature일 때만 matching
        
        KeyPoint pt2 = keypoints2[nn];
        srcPoints.push_back(pt1.pt);
        dstPoints.push_back(pt2.pt);
    }
}

template <typename T>
Mat cal_affine(int ptl_x[], int ptl_y[], int ptr_x[], int ptr_y[], int number_of_points) {
    
    //Mx=b
    Mat M(2 * number_of_points, 6, CV_32F, Scalar(0));
    Mat b(2 * number_of_points, 1, CV_32F);

    Mat M_trans, temp, affineM;                             //affineM = x = A21 or A12

    // initialize matrix
    for (int i = 0; i < number_of_points; i++) {    //행 돌아가는중
        M.at<T>(2 * i, 0) = ptl_x[i];            M.at<T>(2 * i, 1) = ptl_y[i];            M.at<T>(2 * i, 2) = 1;
        M.at<T>(2 * i + 1, 3) = ptl_x[i];        M.at<T>(2 * i + 1, 4) = ptl_y[i];        M.at<T>(2 * i + 1, 5) = 1;
        b.at<T>(2 * i) = ptr_x[i];        b.at<T>(2 * i + 1) = ptr_y[i];
    } //위에 쓴대로 값 집어넣기

    // (M^T * M)^(−1) * M^T * b ( * : Matrix multiplication)
    transpose(M, M_trans);                          //M_trans = M^T
    invert(M_trans * M, temp);                      //temp = (M^T * M)^(−1)
    affineM = temp * M_trans * b;                   //affineM = (M^T * M)^(−1) * M^T * b

    return affineM;
}

void blend_stitching(const Mat I1, const Mat I2, Mat &I_f, int bound_l, int bound_u, float alpha) {

    // I2 is already in I_f by inverse warping
    // I2'는 이미 I_f에 있으니, I1만 고려. I1 자리에 아무것도 없다면 I1값 그대로 옮겨주고, I2가 있다면 0.5비율로 I1과 I2' blending
    for (int i = 0; i < I1.rows; i++) {
        for (int j = 0; j < I1.cols; j++) {                     //I1의 자리만 고려
            bool cond_I2 = I_f.at<Vec3f>(i - bound_u, j - bound_l) != Vec3f(0, 0, 0) ? true : false;    //(0, 0, 0)=검정색이라면 cond_I2=false

            if (cond_I2)    //검정색 아님= I2'있다 => I1과 I2' 겹치는 부분
                I_f.at<Vec3f>(i - bound_u, j - bound_l) = alpha * I1.at<Vec3f>(i, j) + (1 - alpha) * I_f.at<Vec3f>(i - bound_u, j - bound_l);   //I'2가 이미 있다면 alpha=0.5의 비율로 둘다 그려줌
                //i,j는 I1의 왼쪽 위 꼭짓점을 원점으로 하지만 I_f는 bound_u과 bound_l 기준으로 원점이 있다. bound_u, bound_l <=0이므로 i,j 에서 빼준다.
            else            //검정색
                I_f.at<Vec3f>(i - bound_u, j - bound_l) = I1.at<Vec3f>(i, j);   //검정에 I1 image 그대로 옮겨줌

        }
    }
}

Mat AffineTransform(Mat input1, Mat input2, vector<Point2f>& srcPoints, vector<Point2f>& dstPoints){
    const float I1_row = input1.rows;
    const float I1_col = input1.cols;
    const float I2_row = input2.rows;
    const float I2_col = input2.cols;
   
    int n = (int)(srcPoints.size());
    int* ptl_x = new int[n];    int* ptl_y = new int[n];
    int* ptr_x = new int[n];    int* ptr_y = new int[n];

        for (int i = 0; i < n; ++i) {
            ptl_x[i] = dstPoints[i].x;
            ptl_y[i] = dstPoints[i].y;
            ptr_x[i] = srcPoints[i].x;
            ptr_y[i] = srcPoints[i].y;
        }


    // calculate affine Matrix A12, A21
    Mat A12 = cal_affine<float>(ptl_x, ptl_y, ptr_x, ptr_y, n);                // [x'; y'] = A12 [x; y; 1]  (x' y':I2의 좌표 , x y:I1의 좌표 )
    Mat A21 = cal_affine<float>(ptr_x, ptr_y, ptl_x, ptl_y, n);                // [x'; y'] = A12 [x; y; 1]  (x' y':I1의 좌표 , x y:I2의 좌표 )

    Point2f p1(A21.at<float>(0) * 0 + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * 0 + A21.at<float>(5));
    Point2f p2(A21.at<float>(0) * 0 + A21.at<float>(1) * I2_row + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * I2_row + A21.at<float>(5));
    Point2f p3(A21.at<float>(0) * I2_col + A21.at<float>(1) * I2_row + A21.at<float>(2), A21.at<float>(3) * I2_col + A21.at<float>(4) * I2_row + A21.at<float>(5));
    Point2f p4(A21.at<float>(0) * I2_col + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * I2_col + A21.at<float>(4) * 0 + A21.at<float>(5));

    int bound_u = (int)round(min(0.0f, min(p1.y, p4.y)));               //I1의 원점(0)과 I2'의 꼭짓점 중 가장 위(y값 작은것)의 값이 I_f(결과물)의 원점의 y좌표가 된다.
    int bound_b = (int)round(max(I1_row-1, max(p2.y, p3.y)));           //I1의 row(=y값)과 I2'의 꼭짓점 중 가장 아래(y값 큰것)의 값이 I_f(결과물)의 바닥이 된다.
    int bound_l = (int)round(min(0.0f, min(p1.x, p2.x)));               //I1의 원점(0)과 I2'의 꼭짓점 중 가장 왼쪽(x값 작은것)의 값이 I_f(결과물)의 원점의 x좌표가 된다.
    int bound_r = (int)round(max(I1_col-1, max(p3.x, p4.x)));           //I1의 col(=x값)과 I2'의 꼭짓점 중 가장 오른쪽(x값 큰것)의 값이 I_f(결과물)의 가장 오른쪽의 경계가 된다.
                                                                        //bound_u, bound_l <=0 이다.
    // initialize merged image
    Mat I_f(bound_b - bound_u + 1, bound_r - bound_l + 1, CV_32FC3, Scalar(0));     //결과물 저장할 행렬

    //I2' 먼저 그려준다
    // inverse warping with bilinear interplolation
    for (int i = bound_u; i <= bound_b; i++) {                          //행
        for (int j = bound_l; j <= bound_r; j++) {                      //열
            //i행j열은 I_f 위의 점. A12를 이용해 ij에 대응하는 I2의 점(x,y)를 찾는다.
            //[x; y] = A12 [j; i; 1]
            float x = A12.at<float>(0) * j + A12.at<float>(1) * i + A12.at<float>(2) - bound_l;     //j가 bound_l에서 시작하니까 bound_l을 빼주어 원점을 0으로 맞춰준다.
            float y = A12.at<float>(3) * j + A12.at<float>(4) * i + A12.at<float>(5) - bound_u;     //i가 bound_u에서 시작하니까 bound_u을 빼주어 원점을 0으로 맞춰준다.

            float y1 = floor(y);    //y 버림. y와 가장 가까운 y보다 작은 정수
            float y2 = ceil(y);     //y 올림. y와 가장 가까운 y보다 큰 정수
            float x1 = floor(x);    //x 버림
            float x2 = ceil(x);     //x 올림

            float mu = y - y1;      //y의 소수부분
            float lambda = x - x1;  //x의 소수부분

            if (x1 >= 0 && x2 < I2_col && y1 >= 0 && y2 < I2_row)   //I2의 범위 안이라면
                //f(x, y') = mu * f(x, y+1) + (1-mu) * f(x, y)
                //f(x+1, y') = mu * f(x+1, y+1) + (1-mu) * f(x+1, y)
                //f(x', y') = lambda * f(x+1, y') + (1-lambda) * f(x, y')
                I_f.at<Vec3f>(i - bound_u, j - bound_l) = lambda * (mu * input2.at<Vec3f>(y2, x2) + (1 - mu) * input2.at<Vec3f>(y1, x2)) +
                                                          (1 - lambda) *(mu * input2.at<Vec3f>(y2, x1) + (1 - mu) * input2.at<Vec3f>(y1, x1));  //bilinear interpolation
        }
    }//input2'그려지고, I1은 아직 없는 상태
    // image stitching with blend
    blend_stitching(input1, input2, I_f, bound_l, bound_u, 0.5);

    return I_f;
}

Mat SIFTfunc(Mat input1, Mat input2,vector<KeyPoint>& keypoints1, Mat& descriptors1,vector<KeyPoint>& keypoints2, Mat& descriptors2){
    Mat input1_gray, input2_gray;
    cvtColor(input1, input1_gray, CV_RGB2GRAY);
    cvtColor(input2, input2_gray, CV_RGB2GRAY);
    
    Size size = input2.size();
    Size sz = Size(size.width + input1_gray.size().width, max(size.height, input1_gray.size().height));
    Mat matchingImage = Mat::zeros(sz, CV_8UC3);

    input1.copyTo(matchingImage(Rect(size.width, 0, input1_gray.size().width, input1_gray.size().height)));
    input2.copyTo(matchingImage(Rect(0, 0, size.width, size.height)));
    
    vector<Point2f> srcPoints;
    vector<Point2f> dstPoints;
    findPairs(keypoints2, descriptors2, keypoints1, descriptors1, srcPoints, dstPoints);   //keypoint를 matching한다
    printf("%zd keypoints are matched.\n", srcPoints.size());
    
    input1.convertTo(input1, CV_32FC3, 1.0 / 255);
    input2.convertTo(input2, CV_32FC3, 1.0 / 255);
    Mat I_f=AffineTransform(input1, input2,srcPoints,dstPoints);
    return I_f;
}
