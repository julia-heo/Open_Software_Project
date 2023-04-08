#include "hist_func.h"  //헤더파일 추가

int main() {
    
    Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);       //"input.jpg" 이미지를 컬러로 불러와서 input 변수에 대입
    Mat input_gray;

    cvtColor(input, input_gray, CV_RGB2GRAY);    // convert RGB to Grayscale
    //input을 흑백으로 바꿔 input_gray 변수에 대입
    
    // PDF, CDF txt files
    FILE *f_PDF, *f_CDF;
    
//    fopen_s(&f_PDF, "PDF.txt", "w+");
//    fopen_s(&f_CDF, "CDF.txt", "w+");
    f_PDF=fopen("PDF.txt", "w+");               //fopen 함수를 사용해 "PDF.txt"를 읽고 쓰기 모드로 연다. 원래 파일이 없으므로 새로운 파일을 생성한다.
    f_CDF=fopen("CDF.txt", "w+");               //fopen 함수를 사용해 "CDF.txt"를 읽고 쓰기 모드로 연다. 원래 파일이 없으므로 새로운 파일을 생성한다.

    // each histogram
    float *PDF = cal_PDF(input_gray);        // PDF of Input image(Grayscale) : [L]
    float *CDF = cal_CDF(input_gray);        // CDF of Input image(Grayscale) : [L]

    for (int i = 0; i < L; i++) {            //0~L(256)(=모든 색)에 해당하는 픽셀의 개수
        // write PDF, CDF
        fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);  //"PDF.txt"에 계산한 PDF값 저장
        fprintf(f_CDF, "%d\t%f\n", i, CDF[i]);  //"CDF.txt"에 계산한 CDF값 저장
    }

    // memory release
    free(PDF);
    free(CDF);
    fclose(f_PDF);
    fclose(f_CDF);
    
    ////////////////////// Show each image ///////////////////////
    
    namedWindow("RGB", WINDOW_AUTOSIZE);
    imshow("RGB", input);

    namedWindow("Grayscale", WINDOW_AUTOSIZE);
    imshow("Grayscale", input_gray);

    //////////////////////////////////////////////////////////////

    waitKey(0);

    return 0;
}
