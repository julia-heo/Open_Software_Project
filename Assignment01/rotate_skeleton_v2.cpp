#include <opencv2/opencv.hpp>               //라이브러리 추가
#include <iostream>

using namespace cv;

template <typename T>                       //함수 템플릿: 자료형을 정하지 않고, 함수를 만들어 둔다
Mat myrotate(const Mat input, float angle, const char* opt);    //rotate한 결과물(Mat자료형)을 반환하는 함수 정의. parameter: input image, 돌릴 각도, rotation option

int main()
{
    Mat input, rotated;             //input, output 이미지를 저장할 행렬 변수 선언
    
    // Read each image
    input = imread("lena.jpg");     //이미지 파일 읽기

    // Check for invalid input
    if (!input.data) {              //이미지 input 실패시 처리
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    
    // original image
    namedWindow("image");           //"image"라는 이름을 갖는 창을 생성
    imshow("image", input);         //input 이미지 띄움

    rotated = myrotate<Vec3b>(input, 45, "bilinear");

    // rotated image
    namedWindow("rotated");
    imshow("rotated", rotated); //"rotated"창에 결과물 rotated 띄운다

    waitKey(0);                 //대기 함수

    return 0;
}

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt) {
    int row = input.rows;
    int col = input.cols;

    float radian = angle * CV_PI / 180;     //세타

    float sq_row = ceil(row * sin(radian) + col * cos(radian)); //새로운 이미지 크기 row
    float sq_col = ceil(col * sin(radian) + row * cos(radian)); //새로운 이미지 크기 col

    Mat output = Mat::zeros(sq_row, sq_col, input.type());      //새 이미지 0으로 초기화

    for (int i = 0; i < sq_row; i++) {              //행
        for (int j = 0; j < sq_col; j++) {          //열
            //inverse warping
            
            //[x'; y'] = [cosθ  -sinθ ; sinθ  cosθ ] [x; y]
            //x'=cosθ x - sinθ y ,  y'=sinθ x + cosθ y
            //x,y는 input image의 좌표. i,j는 rotate후의 좌표
            //output image의 (j,i)의 데이터를 알아내기 위해 input image의 (x,y)를 계산해 알아낸다. 계산된 (x,y)의 데이터를 (j,i)에 복사한다
                // i행 j열 -> 좌표 (j,i)
            float x = (j - sq_col / 2) * cos(radian) - (i - sq_row / 2) * sin(radian) + col / 2;
            //rotate된 이미지의 좌표에  sq_col / 2, sq_row / 2 뺀다: rotate된 이미지의 중앙이 원점
            //cosθ x - sinθ y에 대입: θ만큼 돌린다. -> 이미지 똑바로 됨, 정 중앙이 원점
            //col / 2를 더한다 : 이미지의 왼쪽 위 꼭짓점이 원점이 된다
            float y = (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;
            //rotate된 이미지의 좌표에  sq_col / 2, sq_row / 2 뺀다: rotate된 이미지의 중앙이 원점
            //sinθ x + cosθ y에 대입: θ만큼 돌린다. -> 이미지 똑바로 됨, 정 중앙이 원점
            //row / 2를 더한다 : 이미지의 왼쪽 위 꼭짓점이 원점이 된다
            
            if ((y >= 0) && (y <= (row - 1)) && (x >= 0) && (x <= (col - 1))) {     //input image의 범위 안의 픽셀이라면
                if (!strcmp(opt, "nearest")) {          // Nearest-neighbor interpolation
                    float xNearest=round(x);            // round() 반올림 함수. x와 가장 가까운 정수. (행렬은 정수로 접근가능)
                    float yNearest=round(y);            // round() 반올림 함수. y와 가장 가까운 정수.

                    output.at<Vec3b>(i,j)=input.at<Vec3b>(yNearest,xNearest);
                    //at 함수는 (행, 열)로 데이터에 접근. input img의 좌표(x,y)에서 가장 가까운 격자점 yNearest행xNearest열의 데이터를 out[ut image i행 j열에 대입

                }
                else if (!strcmp(opt, "bilinear")) {    //Bilinear interpolation
                    float mu=x-(int)x;                  //mu는 x의 소수부분
                    float lambda=y-(int)y;              //lambda는 y의 소수부분
                    
                    //f(x, y') = mu * f(x, y+1) + (1-mu) * f(x, y)
                    //f(x+1, y') = mu * f(x+1, y+1) + (1-mu) * f(x+1, y)
                    //f(x', y') = lambda * f(x+1, y') + (1-lambda) * f(x, y')
                    output.at<Vec3b>(i,j)=(1-lambda)*(mu*input.at<Vec3b>(y,x+1)+(1-mu)*input.at<Vec3b>(y,x))+lambda*(mu*input.at<Vec3b>(y+1,x+1)+(1-mu)*input.at<Vec3b>(y+1,x));

                }
            }
        }
    }

    return output;
}
