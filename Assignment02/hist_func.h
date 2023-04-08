#pragma once

#include <opencv2/opencv.hpp>
#include <stdio.h>

#define L 256        // # of intensity levels
#define IM_TYPE    CV_8UC3

using namespace cv;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

// generate PDF for single channel image
float *cal_PDF(Mat &input) {

    int count[L] = { 0 };       //각 인덱스에 해당하는 색 데이터의 픽셀 개수 저장할 배열
    float *PDF = (float*)calloc(L, sizeof(float));  //float L개 크기만큼 동적 메모리 할당

    // Count
    for (int i = 0; i < input.rows; i++)        //입력 이미지 행
        for (int j = 0; j < input.cols; j++)    //열
            count[input.at<G>(i, j)]++;         //색 데이터에 해당하는 배열 인텍스에 접근해 +1
    //count[]에 색상별 픽셀 개수 저장 완료
    
    // Compute PDF
    for (int i = 0; i < L; i++)
        PDF[i] = (float)count[i] / (float)(input.rows * input.cols);    //PDF는 모든 데이터의 합이 1이므로, 전체 픽셀 수로 count[i]를 나눠준다. 이를 PDF에 저장한다

    return PDF;     //PDF 반환. PDF[0]~PDF[256]에 PDF 데이터 저장됨
}

// generate PDF for color image
float **cal_PDF_RGB(Mat &input) {

    int count[L][3] = { 0 };                            //색 L이 사용된 픽셀 수 저장할 배열 [0]: B, [1]: G, [2]: R
    float **PDF = (float**)malloc(sizeof(float*) * L);  //float L개 크기만큼 동적 메모리 할당

    for (int i = 0; i < L; i++)
        PDF[i] = (float*)calloc(3, sizeof(float));      //float 크기의 변수를 3개 저장할 수 있는 공간 할당
    for (int i = 0; i < input.rows; i++){        //입력 이미지 행
        for (int j = 0; j < input.cols; j++){    //열
            //count[i][j]는, (i,j)의 픽셀의 데이터중 j(=RGB중 하나)의 값이 i라는 의미이다
            count[input.at<Vec3b>(i, j)[0]][0]++;
            count[input.at<Vec3b>(i, j)[1]][1]++;
            count[input.at<Vec3b>(i, j)[2]][2]++;
            //input.at<Vec3b>(i, j)[k] : input의 (i,j)픽셀에 접근한다. k색의 데이터를 가져온다
            //k색에 그 값이 하나 있다는 것을 알게된 것이므로 count[가져온 색 데이터][k]에 +1해준다
        }
    }
    
    for (int i = 0; i < L; i++){    //
        PDF[i][0] = (float)count[i][0] / (float)(input.rows * input.cols);
        PDF[i][1] = (float)count[i][1] / (float)(input.rows * input.cols);
        PDF[i][2] = (float)count[i][2] / (float)(input.rows * input.cols);
        //PDF는 모든 데이터의 합이 1이므로, 전체 픽셀 수로 count를 나눠준다
        //각 색별로 각자의 PDF를 가진다
    }

    return PDF;
}

// generate CDF for single channel image
float *cal_CDF(Mat &input) {

    int count[L] = { 0 };          //각 인덱스에 해당하는 색 데이터의 픽셀 개수 저장할 배열
    float *CDF = (float*)calloc(L, sizeof(float));  //float L개 크기만큼 동적 메모리 할당

    // Count
    for (int i = 0; i < input.rows; i++)        //행
        for (int j = 0; j < input.cols; j++)    //열
            count[input.at<G>(i, j)]++; //색 데이터에 해당하는 배열 인텍스에 접근해 +1
    //count[]에 색상별 픽셀 개수 저장 완료
    
    // Compute CDF
    for (int i = 0; i < L; i++) {
        CDF[i] = (float)count[i] / (float)(input.rows * input.cols); //우선 i색에 해당하는 PDF를 구한다. PDF는 모든 데이터의 합이 1이므로, 전체 픽셀 수로 count[i]를 나눠준다.

        if (i != 0)
            CDF[i] += CDF[i - 1];   //CDF는 누적분포함수이므로 이전 인덱스의 cdf를 더해준다.
    }

    return CDF; //CDF 반환. CDF[0]~CDF[256]에 CDF 데이터 저장됨
}

// generate CDF for color image
float **cal_CDF_RGB(Mat &input) {

    int count[L][3] = { 0 };                //색 데이터의 픽셀 개수 저장할 배열 [0]: B, [1]: G, [2]: R
    float **CDF = (float**)malloc(sizeof(float*) * L);  //float L개 크기만큼 동적 메모리 할당

    for (int i = 0; i < L; i++)
        CDF[i] = (float*)calloc(3, sizeof(float));      //CDF의 각 요소에 float 3개만큼의 메모리를 할당한다
                                                        //CDF[L][3] 같이 사용할 수 있다

    for (int i = 0; i < input.rows; i++){        //입력 이미지 행
        for (int j = 0; j < input.cols; j++){    //열
            //count[i][j]는, (i,j)의 픽셀의 데이터중 j(=RGB중 하나)의 값이 i라는 의미이다
            count[input.at<Vec3b>(i, j)[0]][0]++;
            count[input.at<Vec3b>(i, j)[1]][1]++;
            count[input.at<Vec3b>(i, j)[2]][2]++;
            //input.at<Vec3b>(i, j)[k] : input의 (i,j)픽셀에 접근한다. k색의 데이터를 가져온다
            //k색에 그 값이 하나 있다는 것을 알게된 것이므로 count[가져온 색 데이터][k]에 +1해준다
        }
    }
    
    for (int i = 0; i < L; i++){
        CDF[i][0] = (float)count[i][0] / (float)(input.rows * input.cols);
        CDF[i][1] = (float)count[i][1] / (float)(input.rows * input.cols);
        CDF[i][2] = (float)count[i][2] / (float)(input.rows * input.cols);
        //우선 각 RGB별로 i색에 해당하는 PDF를 구한다.
        //PDF는 모든 데이터의 합이 1이므로, 전체 픽셀 수로 count를 나눠준다.
        
        if (i != 0){
            CDF[i][0] += CDF[i - 1][0];
            CDF[i][1] += CDF[i - 1][1];
            CDF[i][2] += CDF[i - 1][2];
            //CDF는 누적분포함수이므로 이전 인덱스의 cdf를 더해준다
        }
        
    }

    return CDF;
}
