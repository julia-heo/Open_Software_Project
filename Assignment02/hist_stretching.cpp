#include "hist_func.h"

void linear_stretching(Mat &input, Mat &stretched, G *trans_func, G x1, G x2, G y1, G y2);  //함수 미리 선언

int main() {

    Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);   //"input.jpg" 이미지를 컬러로 불러와서 input 변수에 대입
    Mat input_gray;

    cvtColor(input, input_gray, CV_RGB2GRAY);    // convert RGB to Grayscale
                                                 //input을 흑백으로 바꿔 input_gray 변수에 대입

    Mat stretched = input_gray.clone();

    // PDF or transfer function txt files
    FILE *f_PDF;
    FILE *f_stretched_PDF;
    FILE *f_trans_func_stretch;

//    fopen_s(&f_PDF, "PDF.txt", "w+");
//    fopen_s(&f_stretched_PDF, "stretched_PDF.txt", "w+");
//    fopen_s(&f_trans_func_stretch, "trans_func_stretch.txt", "w+");
    f_PDF=fopen("PDF.txt", "w+");                                   //fopen 함수를 사용해 "PDF.txt"를 읽고 쓰기 모드로 연다. 원래 파일이 없으므로 새로운 파일을 생성한다.
    f_stretched_PDF=fopen("stretched_PDF.txt", "w+");               //fopen 함수를 사용해 "stretched_PDF.txt"를 읽고 쓰기 모드로 연다. 원래 파일이 없으므로 새로운 파일을 생성한다.
    f_trans_func_stretch=fopen("trans_func_stretch.txt", "w+");     //fopen 함수를 사용해 "trans_func_stretch.txt"를 읽고 쓰기 모드로 연다. 원래 파일이 없으므로 새로운 파일을 생성한다.

    G trans_func_stretch[L] = { 0 };             //헤더파일에서 typedef로 선언된 G는 uchar.linear stretching function을 적용한 값(=mapping될 색 데이터 값)

    float *PDF = cal_PDF(input_gray);           //PDF 만든다. 각 인덱스에 해당하는 색 데이터의 픽셀 개수의 비율이 저장된다.

    linear_stretching(input_gray, stretched, trans_func_stretch, 50, 110, 10, 110);    // histogram stretching (x1 ~ x2 -> y1 ~ y2)
                    //매개변수:(입력된 이미지, stretching 결과를 넣을 Mat, mapping될 색 저장할 배열, x1,x2(pixel몰린 구간),y1,y2(pixel을 집중적으로 나눠주고싶은 구간))
    float *stretched_PDF = cal_PDF(stretched);                                        // stretched PDF

    for (int i = 0; i < L; i++) {
        // write PDF
        fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);      //input image 의 PDF 저장
        fprintf(f_stretched_PDF, "%d\t%f\n", i, stretched_PDF[i]);  //stretch된 histogram(PDF)를 "trans_func_stretch.txt"에 저장

        // write transfer functions
        fprintf(f_trans_func_stretch, "%d\t%d\n", i, trans_func_stretch[i]);    //각 색 데이터가 mapping된 색 저장
    }
    
    // memory release
    free(PDF);
    free(stretched_PDF);
    fclose(f_PDF);
    fclose(f_stretched_PDF);
    fclose(f_trans_func_stretch);
    
    ////////////////////// Show each image ///////////////////////

    namedWindow("Grayscale", WINDOW_AUTOSIZE);
    imshow("Grayscale", input_gray);    //input한 흑백 이미지 새로운 창에 띄우기

    namedWindow("Stretched", WINDOW_AUTOSIZE);      //stretching한 이미지 새로운 창에 띄우기
    imshow("Stretched", stretched);

    //////////////////////////////////////////////////////////////

    waitKey(0);

    return 0;
}

// histogram stretching (linear method)
void linear_stretching(Mat &input, Mat &stretched, G *trans_func, G x1, G x2, G y1, G y2) {

    float constant = (y2 - y1) / (float)(x2 - x1);      //x1~x2구간의 linear stretching function의 기울기 계산

    // compute transfer function
    for (int i = 0; i < L; i++) {   //모든 색
        if (i >= 0 && i <= x1)                                  //x0~x1 구간: (0,0)과 (x1,y1)연결
            trans_func[i] = (G)(y1 / x1 * i);                   //i를 (y1-0) / (x1-0)를 기울기로 하는 linear function에 대입
        else if (i > x1 && i <= x2)                             //x1~x2 구간: (x1,y1)과 (x2,y2)연결
            trans_func[i] = (G)(constant * (i - x1) + y1);      //constant를 기울기로 하고 (x1,y1)을 지나는 linear function에 i를 대입
        else                                                    //x2~x3 구간: (x2,y2)과 (x3,y3)연결
            trans_func[i] = (G)((L - 1 - x2) / (L - 1 - y2) * (i - x2) + y2);   //(L - 1 - x2)/(L - 1 - y2)를 기울기로 하고 (x2,y2)를 지나는 linear function에 i 대입
    }   //i의 색 데이터 값을 trans_func[i]에 mapping시켜 stretching 해줄것임

    // perform the transfer function
    for (int i = 0; i < input.rows; i++)            //행
        for (int j = 0; j < input.cols; j++)        //열
            stretched.at<G>(i, j) = trans_func[input.at<G>(i, j)];  //input image의 색상 값을 trans_func에 넣고 계산한 값에 mapping
}
