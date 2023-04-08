#include "hist_func.h"

void hist_eq_Color(Mat &input, Mat &equalized, G(*trans_func)[3], float **CDF);

int main() {

    Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);   //"input.jpg" 이미지를 컬러로 불러와서 input 변수에 대입
    Mat equalized_RGB = input.clone();          //equalized_RGB에 input 이미지 복사

    // PDF or transfer function txt files
    FILE *f_equalized_PDF_RGB, *f_PDF_RGB;
    FILE *f_trans_func_eq_RGB;
    
//    fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+");
//    fopen_s(&f_equalized_PDF_RGB, "equalized_PDF_RGB.txt", "w+");
//    fopen_s(&f_trans_func_eq_RGB, "trans_func_eq_RGB.txt", "w+");
    f_PDF_RGB=fopen("PDF_RGB.txt", "w+");                       //"PDF_RGB.txt" 파일 생성후 읽기, 쓰기 모드로 연다
    f_equalized_PDF_RGB=fopen("equalized_PDF_RGB.txt", "w+");   //"equalized_PDF_RGB.txt" 파일 생성후 읽기, 쓰기 모드로 연다
    f_trans_func_eq_RGB=fopen("trans_func_eq_RGB.txt", "w+");   //"trans_func_eq_RGB.txt" 파일 생성후 읽기, 쓰기 모드로 연다
    
    //컬러 이미지이므로, RGB를 구분해 저장하기 위해 이차원 배열을 사용한다.
    //인덱스 0:B 1:G 2:R 이다.
    float **PDF_RGB = cal_PDF_RGB(input);    // PDF of Input image(RGB) : [L][3]
    float **CDF_RGB = cal_CDF_RGB(input);    // CDF of Input image(RGB) : [L][3]

    G trans_func_eq_RGB[L][3] = { 0 };        // transfer function
    //trans_func_eq_RGB[L][i]=k -> i색(RGB중 하나)의 데이터 L이 매칭될 색 k

    hist_eq_Color(input, equalized_RGB, trans_func_eq_RGB, CDF_RGB);    //histogram equalization 수행
    // histogram equalization on RGB image
    
    float **equalized_PDF_RGB=cal_PDF_RGB(equalized_RGB);               //histogram equalization 완료한 이미지의 PDF 구하기
    // equalized PDF (RGB)
    
    
    for (int i = 0; i < L; i++) {
        // write PDF
        fprintf(f_PDF_RGB, "%d\t%f %f %f\n", i, PDF_RGB[i][0],PDF_RGB[i][1],PDF_RGB[i][2]); //input image 의 PDF를 "PDF_RGB.txt"에 저장
        fprintf(f_equalized_PDF_RGB, "%d\t%f %f %f\n",i,equalized_PDF_RGB[i][0],equalized_PDF_RGB[i][1],equalized_PDF_RGB[i][2]); //Histogram Equalization한 이미지의 PDF를 "equalized_PDF_RGB.txt"에 저장
        
        // write transfer functions
        fprintf(f_trans_func_eq_RGB, "%d\t%d %d %d\n",i,trans_func_eq_RGB[i][0],trans_func_eq_RGB[i][1],trans_func_eq_RGB[i][2]); //색 데이터가 i라면 각 색별로 어떤 데이터에 매칭되는지 "trans_func_eq_RGB.txt"에 저장.
    }

    // memory release
    free(PDF_RGB);
    free(CDF_RGB);
    fclose(f_PDF_RGB);
    fclose(f_equalized_PDF_RGB);
    fclose(f_trans_func_eq_RGB);

    ////////////////////// Show each image ///////////////////////

    namedWindow("RGB", WINDOW_AUTOSIZE);
    imshow("RGB", input);                   //input한 이미지 새로운 창에 띄우기

    namedWindow("Equalized_RGB", WINDOW_AUTOSIZE);
    imshow("Equalized_RGB", equalized_RGB); //Histogram Equalization 완료한 이미지 새로운 창에 띄우기

    //////////////////////////////////////////////////////////////

    waitKey(0);

    return 0;
}

// histogram equalization on 3 channel image
void hist_eq_Color(Mat &input, Mat &equalized, G(*trans_func)[3], float **CDF) {
    //매개 변수:(input 이미지, equalization의 결과 저장할 행렬, mapping된 색 저장할 배열, CDF)
    
    for (int i = 0; i < L; i++){
        //Histogram Equalization의 식: T(r)=(L-1) CDF(r)
        //컬러이미지이므로 각각 R G B에 각각 계산 적용
        trans_func[i][0] = (G)((L - 1) * CDF[i][0]);
        trans_func[i][1] = (G)((L - 1) * CDF[i][1]);
        trans_func[i][2] = (G)((L - 1) * CDF[i][2]);
        //색 i가 mapping될 색 trans_func[i][]에 저장 완료
    }
    for (int i = 0; i < input.rows; i++){
        for (int j = 0; j < input.cols; j++){
            // access multi channel matrix element:
            // if matrix A is CV_8UC3 type, A(i, j, k) -> A.at<Vec3b>(i, j)[k]
            equalized.at<Vec3b>(i, j)[0] = trans_func[input.at<Vec3b>(i, j)[0]][0];
            equalized.at<Vec3b>(i, j)[1] = trans_func[input.at<Vec3b>(i, j)[1]][1];
            equalized.at<Vec3b>(i, j)[2] = trans_func[input.at<Vec3b>(i, j)[2]][2];
            //input의 (i,j)에 접근해 구한 색 데이터를 trans_func[]에 넣어 어느 색으로 바뀔지 알아내어 equalized의 (i,j)에 대입한다
            //각 RGB에 대해 따로따로 적용
        }
    }
}
