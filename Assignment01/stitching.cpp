// stitching.cpp : Defines the entry point for the console application.
//

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

template <typename T>                                   //함수 템플릿: 자료형을 정하지 않고, 함수를 만들어 둔다
Mat cal_affine(int ptl_x[], int ptl_y[], int ptr_x[], int ptr_y[], int number_of_points);

void blend_stitching(const Mat I1, const Mat I2, Mat &I_f, int diff_x, int diff_y, float alpha);

int main() {
    Mat I1, I2;
 
    // Read each image
    I1 = imread("stitchingL.jpg");
    I2 = imread("stitchingR.jpg");

    I1.convertTo(I1, CV_32FC3, 1.0 / 255);  //값을 CV_32FC3(32비트 float)로 전환, 데이터(색 정보)를 255로 나누어 (0.0 ~ 1.0) 사이의 값으로 변경
    I2.convertTo(I2, CV_32FC3, 1.0 / 255);
    
    // corresponding pixels
    int ptl_x[28] = { 509, 558, 605, 649, 680, 689, 705, 730, 734, 768, 795, 802, 818, 837, 877, 889, 894, 902, 917, 924, 930, 948, 964, 969, 980, 988, 994, 998 };
    int ptl_y[28] = { 528, 597, 581, 520, 526, 581, 587, 496, 506, 500, 342, 558, 499, 642, 474, 456, 451, 475, 530, 381, 472, 475, 426, 539, 329, 341, 492, 511 };
    int ptr_x[28] = { 45, 89, 142, 194, 226, 230, 246, 279, 281, 314, 352, 345, 365, 372, 421, 434, 439, 446, 456, 472, 471, 488, 506, 503, 527, 532, 528, 531 };
    int ptr_y[28] = { 488, 561, 544, 482, 490, 546, 552, 462, 471, 467, 313, 526, 468, 607, 445, 429, 424, 447, 500, 358, 446, 449, 403, 510, 312, 324, 466, 484 };

    // Check for invalid input
    if (!I1.data || !I2.data) {             //input 안됐을 때 처리
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // height(row), width(col) of each image
    const float I1_row = I1.rows;
    const float I1_col = I1.cols;
    const float I2_row = I2.rows;
    const float I2_col = I2.cols;

    // calculate affine Matrix A12, A21
    Mat A12 = cal_affine<float>(ptl_x, ptl_y, ptr_x, ptr_y, 28);                // [x'; y'] = A12 [x; y; 1]  (x' y':I2의 좌표 , x y:I1의 좌표 )
    Mat A21 = cal_affine<float>(ptr_x, ptr_y, ptl_x, ptl_y, 28);                // [x'; y'] = A12 [x; y; 1]  (x' y':I1의 좌표 , x y:I2의 좌표 )

    // compute corners (p1, p2, p3, p4)
    //(행, 열)
    // p1: (0,0)
    // p2: (row, 0)
    // p3: (row, col)
    // p4: (0, col)
    //A21=[a b c; d e f]이라 하자. [x'; y'] = A21 [x; y; 1] , x'=ax+by+c, y'=dx+ey+f
    //cal_affine를 통해 A21를 구해 a=A21.at<float>(0), b=A21.at<float>(1) ... f=A21.at<float>(5)
    //이를 대입해 I2의 네 꼭짓점을 이동한 P1~P4의 좌표 (x',y')를 구하자
    //위의 p1~p4 좌표는 (행,렬)으로 (y좌표, x좌표)이다. 이를 맞춰 대입하자
    Point2f p1(A21.at<float>(0) * 0 + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * 0 + A21.at<float>(5));
    Point2f p2(A21.at<float>(0) * 0 + A21.at<float>(1) * I2_row + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * I2_row + A21.at<float>(5));
    Point2f p3(A21.at<float>(0) * I2_col + A21.at<float>(1) * I2_row + A21.at<float>(2), A21.at<float>(3) * I2_col + A21.at<float>(4) * I2_row + A21.at<float>(5));
    Point2f p4(A21.at<float>(0) * I2_col + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * I2_col + A21.at<float>(4) * 0 + A21.at<float>(5));

    // compute boundary for merged image(I_f)
    // bound_u <= 0
    // bound_b >= I1_row-1
    // bound_l <= 0
    // bound_b >= I1_col-1
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
                I_f.at<Vec3f>(i - bound_u, j - bound_l) = lambda * (mu * I2.at<Vec3f>(y2, x2) + (1 - mu) * I2.at<Vec3f>(y1, x2)) +
                                                          (1 - lambda) *(mu * I2.at<Vec3f>(y2, x1) + (1 - mu) * I2.at<Vec3f>(y1, x1));  //bilinear interpolation
        }
    }//I2'그려지고, I1은 아직 없는 상태

    // image stitching with blend
    blend_stitching(I1, I2, I_f, bound_l, bound_u, 0.5);
    
    //"Left Image"창에 I1 이미지 띄우기
    namedWindow("Left Image");
    imshow("Left Image", I1);

    //"Right Image"창에 I2 이미지 띄우기
    namedWindow("Right Image");
    imshow("Right Image", I2);

    //"result"창에 I_f(결과물) 이미지 띄우기
    namedWindow("result");
    imshow("result", I_f);

    I_f.convertTo(I_f, CV_8UC3, 255.0);
    imwrite("result.png", I_f);
    //결과물 파일로 저장
    
    waitKey(0);

    return 0;
}

template <typename T>
Mat cal_affine(int ptl_x[], int ptl_y[], int ptr_x[], int ptr_y[], int number_of_points) {
    
    //Mx=b 만들것이다.
    //M=[x1 y1 1 0 0 0; 0 0 0 x1 y1 1; x2 y2 1 0 0 0; 0 0 0 x2 y2 1; ... ; x28 y28 1 0 0 0; 0 0 0 x28 y28 1]
    //x=[a; b; c; d; e; f] : 구하고 싶은 Matrix (A21, A12)
    //b=[x1'; y1'; x2'; y2'; ... ; x28'; y28']

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

    int col = I_f.cols;
    int row = I_f.rows;

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
