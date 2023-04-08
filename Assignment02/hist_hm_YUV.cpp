#include "hist_func.h"

void hist_ma(Mat &input, Mat &reference, Mat &equalized, G *trans_func, float *CDF,float *CDF_reference);

int main() {

    Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR); //"input.jpg" 이미지를 컬러로 불러와서 input 변수에 대입
    Mat reference = imread("reference.jpeg", CV_LOAD_IMAGE_COLOR);  //"reference.jpeg"불러와서 reference 변수에 저장

    Mat matched_YUV;
    cvtColor(input, matched_YUV, CV_RGB2YUV);    // RGB -> YUV
    
    Mat reference_YUV;
    cvtColor(reference, reference_YUV, CV_RGB2YUV);    // RGB -> YUV
    
    
    // split each channel(Y, U, V)
    Mat channels[3];
    split(matched_YUV, channels);             //matched_YUV를 색 채널별로 분리해 저장
    Mat Y = channels[0];                        // U = channels[1], V = channels[2]
    
    Mat channels_reference[3];
    split(reference_YUV, channels_reference);   //reference image를 YUV 채널별로 나눠 channels_reference에 저장
    Mat Y_reference = channels_reference[0];    //reference image의 Y 채널의 데이터

    // PDF or transfer function txt files
    FILE *f_matched_PDF_YUV, *f_PDF_RGB;
    FILE *f_trans_func_ma_YUV;

    float **PDF_RGB = cal_PDF_RGB(input);        // PDF of Input image(RGB) : [L][3]
    float *CDF_YUV = cal_CDF(Y);                // CDF of Y channel image

    float *CDF_REF_YUV = cal_CDF(Y_reference);

//    fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+");
//    fopen_s(&f_equalized_PDF_YUV, "equalized_PDF_YUV.txt", "w+");
//    fopen_s(&f_trans_func_eq_YUV, "trans_func_eq_YUV.txt", "w+");
    f_PDF_RGB=fopen("PDF_RGB.txt", "w+");
    f_matched_PDF_YUV=fopen("matched_PDF_YUV.txt", "w+");
    f_trans_func_ma_YUV=fopen("trans_func_ma_YUV.txt", "w+");

    G trans_func_ma_YUV[L] = { 0 };            // transfer function
                                               // Y가 매칭될 수 저장

    // histogram equalization on Y channel
    //hist_ma(Y,channels[0],trans_func_ma_YUV,CDF_YUV); //Apply the HM for Y channel only
    hist_ma(Y,Y_reference,channels[0],trans_func_ma_YUV,CDF_YUV,CDF_REF_YUV); //Apply the HM for Y channel only
    
    // merge Y, U, V channels
    merge(channels, 3, matched_YUV);
    
    // YUV -> RGB (use "CV_YUV2RGB" flag)
    cvtColor(matched_YUV,matched_YUV, CV_YUV2RGB);  //Y만 HM를 적용한 이미지를 RGB로 변환해준다

    // equalized PDF (YUV)
    float **matched_PDF_YUV=cal_PDF_RGB(matched_YUV);   //HM한 이미지 PDF구하기. RGB각각.

    for (int i = 0; i < L; i++) {
        // write PDF
        fprintf(f_PDF_RGB, "%d\t%f %f %f\n", i, PDF_RGB[i][0],PDF_RGB[i][1],PDF_RGB[i][2]); //input image 의 PDF를 "PDF_RGB.txt"에 저장
        fprintf(f_matched_PDF_YUV, "%d\t%f %f %f\n",i,matched_PDF_YUV[i][0],matched_PDF_YUV[i][1],matched_PDF_YUV[i][2]); //HM한 이미지의 PDF를 "matched_PDF_YUV.txt"에 저장
        
        // write transfer functions
        fprintf(f_trans_func_ma_YUV, "%d\t%d\n",i,trans_func_ma_YUV[i]); //Y 채널의 값 i가 어떤 수로 매칭되는지 "trans_func_ma_RGB.txt"에 저장.
    }

    // memory release
    free(PDF_RGB);
    free(CDF_YUV);
    fclose(f_PDF_RGB);
    fclose(f_matched_PDF_YUV);
    fclose(f_trans_func_ma_YUV);

    ////////////////////// Show each image ///////////////////////

    namedWindow("RGB", WINDOW_AUTOSIZE);
    imshow("RGB", input);                           //input한 이미지 새로운 창에 띄우기
    
    namedWindow("Ref", WINDOW_AUTOSIZE);
    imshow("Ref", reference);                           //input한 reference 이미지 새로운 창에 띄우기

    namedWindow("Matched_YUV", WINDOW_AUTOSIZE);
    imshow("Matched_YUV", matched_YUV);         //Y에 HM 완료한 이미지 새로운 창에 띄우기

    //////////////////////////////////////////////////////////////

    waitKey(0);

    return 0;
}

// histogram equalization
void hist_ma(Mat &input, Mat &reference, Mat &equalized, G *trans_func, float *CDF,float *CDF_reference) {

    G trans_func_G_reverse[L] = { 0 };  //G^(-1)를 저장할 배열
    G trans_func_G;
    
    for (int i = 0; i < L; i++){
        trans_func_G=(G)((L - 1) * CDF_reference[i]);   //레퍼런스 이미지의 T(r)
        trans_func_G_reverse[trans_func_G]=i;           //역함수이므로 결과값과 인덱스를 반대로 하여 저장
    }

    
    for (int i = 0; i < L; i++){
        //역함수에서, 연결안된(=값이 0인) 곳을 이전 인덱스의 값으로 채워준다
        if(trans_func_G_reverse[i]==0){
                trans_func_G_reverse[i]=trans_func_G_reverse[i-1];
        }
    }
    
    // compute transfer function
    for (int i = 0; i < L; i++)
        trans_func[i] = (G)((L - 1) * CDF[i]);
        //각 색 Y가 mapping될 값 trans_func[i]에 저장 완료

    // perform the transfer function
    for (int i = 0; i < input.rows; i++){
        for (int j = 0; j < input.cols; j++){
            equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];          //입력 이미지 r에 대해 HE의 함수 T(r) 계산
            equalized.at<G>(i, j) =trans_func_G_reverse[equalized.at<G>(i, j)]; //T(r)을 다시 G^(-1)에 넣어 z를 구한다. z=G^-1(s)
        }
    }
        
            
    
    
    
}
