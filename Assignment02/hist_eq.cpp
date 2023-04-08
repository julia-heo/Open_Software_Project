#include "hist_func.h"

void hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF); //함수 선언

int main() {

    Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);   //"input.jpg" 이미지를 컬러로 불러와서 input 변수에 대입
    Mat input_gray;

    cvtColor(input, input_gray, CV_RGB2GRAY);    // convert RGB to Grayscale
    ////input을 흑백으로 바꿔 input_gray 변수에 대입

    Mat equalized = input_gray.clone();          //equalized 행렬에 input_gray 복사해 대입

    // PDF or transfer function txt files
    FILE *f_PDF;
    FILE *f_equalized_PDF_gray;
    FILE *f_trans_func_eq;
    
//    fopen_s(&f_PDF, "PDF.txt", "w+");
//    fopen_s(&f_equalized_PDF_gray, "equalized_PDF_gray.txt", "w+");
//    fopen_s(&f_trans_func_eq, "trans_func_eq.txt", "w+");
    f_PDF=fopen("PDF.txt", "w+");
    f_equalized_PDF_gray=fopen("equalized_PDF_gray.txt", "w+");
    f_trans_func_eq=fopen("trans_func_eq.txt", "w+");
    //fopen 함수를 사용해 텍스트파일을 읽고 쓰기 모드로 연다. 원래 파일이 없으므로 새로운 파일을 생성한다.
    
    float *PDF = cal_PDF(input_gray);    // PDF of Input image(Grayscale) : [L]
    float *CDF = cal_CDF(input_gray);    // CDF of Input image(Grayscale) : [L]

    G trans_func_eq[L] = { 0 };            // transfer function
    //기존의 색이 mapping될 색 데이터 값을 저장할 배열

    hist_eq(input_gray, equalized, trans_func_eq, CDF); // histogram equalization on grayscale image
    float *equalized_PDF_gray = cal_PDF(equalized);     // equalized PDF (grayscale)
                                                        // equalization 완료한 이미지의 pdf구하기

    for (int i = 0; i < L; i++) {
        // write PDF
        fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);          //input image 의 PDF를 "PDF.txt"에 저장
        fprintf(f_equalized_PDF_gray, "%d\t%f\n", i, equalized_PDF_gray[i]); //Histogram Equalization한 이미지의 PDF를 "equalized_PDF_gray.txt"에 저장

        // write transfer functions
        fprintf(f_trans_func_eq, "%d\t%d\n", i, trans_func_eq[i]);  //Histogram Equalization를 s=T(r)라 할 때, r과 s 순서쌍을 "trans_func_eq.txt"에 저장
    }

    // memory release
    free(PDF);
    free(CDF);
    fclose(f_PDF);
    fclose(f_equalized_PDF_gray);
    fclose(f_trans_func_eq);

    ////////////////////// Show each image ///////////////////////

    namedWindow("Grayscale", WINDOW_AUTOSIZE);
    imshow("Grayscale", input_gray);        //input한 흑백 이미지 새로운 창에 띄우기

    namedWindow("Equalized", WINDOW_AUTOSIZE);
    imshow("Equalized", equalized);         //Histogram Equalization 완료한 이미지 새로운 창에 띄우기

    //////////////////////////////////////////////////////////////

    waitKey(0);

    return 0;
}

// histogram equalization
void hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF) {
    //매개 변수:(input 이미지, equalization의 결과 저장할 행렬, mapping된 색 저장할 배열, CDF)
    
    // compute transfer function
    //Histogram Equalization의 식: T(r)=(L-1) CDF(r)
    for (int i = 0; i < L; i++)
        trans_func[i] = (G)((L - 1) * CDF[i]);
    //각 색 i가 mapping될 색 trans_func[i]에 저장 완료
    
    // perform the transfer function
    for (int i = 0; i < input.rows; i++)        //행
        for (int j = 0; j < input.cols; j++)    //열
            equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
            //input의 (i,j)에 접근해 구한 색 데이터를 trans_func[]에 넣어 어느 색으로 바뀔지 알아내어 equalized의 (i,j)에 대입한다
}
