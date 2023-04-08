#include "hist_func.h"

void hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF);

int main() {

    Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR); //"input.jpg" 이미지를 컬러로 불러와서 input 변수에 대입
    Mat equalized_YUV;

    cvtColor(input, equalized_YUV, CV_RGB2YUV);    // RGB -> YUV
    
    // split each channel(Y, U, V)
    Mat channels[3];
    split(equalized_YUV, channels);             //equalized_YUV를 색 채널별로 분리해 저장
    Mat Y = channels[0];                        // U = channels[1], V = channels[2]

    // PDF or transfer function txt files
    FILE *f_equalized_PDF_YUV, *f_PDF_RGB;
    FILE *f_trans_func_eq_YUV;

    float **PDF_RGB = cal_PDF_RGB(input);        // PDF of Input image(RGB) : [L][3]
    float *CDF_YUV = cal_CDF(Y);                // CDF of Y channel image

//    fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+");
//    fopen_s(&f_equalized_PDF_YUV, "equalized_PDF_YUV.txt", "w+");
//    fopen_s(&f_trans_func_eq_YUV, "trans_func_eq_YUV.txt", "w+");
    f_PDF_RGB=fopen("PDF_RGB.txt", "w+");
    f_equalized_PDF_YUV=fopen("equalized_PDF_YUV.txt", "w+");
    f_trans_func_eq_YUV=fopen("trans_func_eq_YUV.txt", "w+");

    G trans_func_eq_YUV[L] = { 0 };            // transfer function
                                               // Y가 매칭될 수 저장

    // histogram equalization on Y channel
    hist_eq(Y,channels[0],trans_func_eq_YUV,CDF_YUV); //Apply the HE for Y channel only
    
    
    // merge Y, U, V channels
    merge(channels, 3, equalized_YUV);
    
    // YUV -> RGB (use "CV_YUV2RGB" flag)
    cvtColor(equalized_YUV,equalized_YUV, CV_YUV2RGB);  //Y만 HE를 적용한 이미지를 RGB로 변환해준다

    // equalized PDF (YUV)
    float **equalized_PDF_YUV=cal_PDF_RGB(equalized_YUV);   //HE한 이미지 PDF구하기. RGB각각.

    for (int i = 0; i < L; i++) {
        // write PDF
        fprintf(f_PDF_RGB, "%d\t%f %f %f\n", i, PDF_RGB[i][0],PDF_RGB[i][1],PDF_RGB[i][2]);
        fprintf(f_equalized_PDF_YUV, "%d\t%f %f %f\n",i,equalized_PDF_YUV[i][0],equalized_PDF_YUV[i][1],equalized_PDF_YUV[i][2]);
        
        // write transfer functions
        fprintf(f_trans_func_eq_YUV, "%d\t%d\n",i,trans_func_eq_YUV[i]);
    }

    // memory release
    free(PDF_RGB);
    free(CDF_YUV);
    fclose(f_PDF_RGB);
    fclose(f_equalized_PDF_YUV);
    fclose(f_trans_func_eq_YUV);

    ////////////////////// Show each image ///////////////////////

    namedWindow("RGB", WINDOW_AUTOSIZE);
    imshow("RGB", input);                           //input한 흑백 이미지 새로운 창에 띄우기

    namedWindow("Equalized_YUV", WINDOW_AUTOSIZE);
    imshow("Equalized_YUV", equalized_YUV);         //Y에 HE 완료한 이미지 새로운 창에 띄우기

    //////////////////////////////////////////////////////////////

    waitKey(0);

    return 0;
}

// histogram equalization
void hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF) {
    //매개 변수:(input 이미지, equalization의 결과 저장할 행렬, mapping된 색 저장할 배열, CDF)
    //Y만 HE하면 된다.

    // compute transfer function
    for (int i = 0; i < L; i++)
        trans_func[i] = (G)((L - 1) * CDF[i]);
        //각 색 Y가 mapping될 값 trans_func[i]에 저장 완료

    // perform the transfer function
    for (int i = 0; i < input.rows; i++)
        for (int j = 0; j < input.cols; j++)
            equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
            //input의 (i,j)에 접근해 구한 데이터를 trans_func[]에 넣어 어느 값으로 바뀔지 알아내어 equalized의 (i,j)에 대입한다
}
