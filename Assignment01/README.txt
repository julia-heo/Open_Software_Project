1. rotate
코드명: rotate_skeleton_v2.cpp
코드의 목적: Nearest-neighbor interpolation과 Bilinear interpolation을 통해 Image Rotation을 c++로 구현
코드가 하는 일: 주어진 이미지("lena.jpg")를 반시계 방향으로 45도 rotate한 결과를 새 창에 띄워 출력한다
실행하는 방법:
	Mac(M1)에서 OpenCV, Xcode 설치 후 사용
변수:
	input: 입력 이미지 파일이 저장된 행렬
	rotated: 결과 이미지 저장할 행렬

함수 소개:
	myrotate(const Mat input, float angle, const char* opt)
		input: 입력 이미지 행렬
		angle: rotate할 각도
		opt: interpolation 옵션. nearest/bilinear 사용가능.
	
		input을 angle 만큼, opt 방식으로 rotate한 행렬 output를 리턴.
		
2. stitching
코드명: stitching.cpp
코드의 목적: 두 이미지 파일(stitchingL.jpg, stitchingR.jpg)에 corresponding pixels이 존재하고, 그 정보가 모두 제공되었다. I_2를 affine transform하여 두 이미지가 이어지도록 합친다.
실행하는 방법: Mac(M1)에서 OpenCV, Xcode 설치 후 사용
변수:
	I1, I2 : 입력 이미지 파일 데이터 저장 행렬
	ptl_x[] , ptl_y[] : corresponding pixels의 왼쪽 이미지에서의 x좌표, y좌표
	ptr_x[] , ptr_y[] : corresponding pixels의 오른쪽 이미지에서의 x좌표, y좌표
	A12 : I1의 좌표에 대응하는 I2위의 점의 좌표를 구할 수 있는 Matrix
	A21 : I2의 좌표에 대응하는 I1위의 점의 좌표를 구할 수 있는 Matrix
	p1 p2 p3 p4 : I1과의 corresponding pixels이 맞도록 I2를 transform 한 I2'의 네 꼭짓점
	bound_u bound_b bound_l bound_r : I1과 I2를 stitching한 결과 이미지의 경계 (위 아래 왼쪽 오른쪽)
	I_f : I1과 I2를 stiching한 결과 이미지를 저장할 행렬

함수:
	cal_affine(int ptl_x[], int ptl_y[], int ptr_x[], int ptr_y[], int number_of_points)
		ptl_x[]: corresponding pixels 의 왼쪽 이미지에서의 x 좌표
		ptl_y[]: corresponding pixels 의 왼쪽 이미지에서의 y 좌표
		ptr_x[]: corresponding pixels 의 오른쪽 이미지에서의 x 좌표 
		ptr_y[]: corresponding pixels 의 오른쪽 이미지에서의 y 좌표 
		number_of_points: corresponding pixels 의 개수

		함수 목적: ptl_x, ptl_y 와 계산해 ptr_x, ptr_y 를 구할 수 있는 Matrix(𝐴12, 𝐴21) 반환

	


	blend_stitching(const Mat I1, const Mat I2, Mat &I_f, int bound_l, int bound_u, float alpha)
		𝐼1: 𝐼1이미지 데이터 담은 행렬. 이 함수를 통해 𝐼_𝑓에 𝐼1데이터 복사할 것이다. 
		𝐼2: 𝐼2이미지 데이터 담은 행렬
		𝐼_𝑓: 결과 이미지 담을 행렬. 현재 𝐼2’만 그려져 있고, 나머지 부분은 검정색 
		bound_l: 𝐼_𝑓의 왼쪽 경계선
		bound_u: 𝐼_𝑓의 위 경계선
		alpha: 𝐼1와 𝐼2 blend 할 비율. 0.5
		
		함수 목적: 𝐼2’만 그려져 있는 𝐼_𝑓에 𝐼1데이터를 추가한다. 𝐼1이미지의 범위에 𝐼2’가 이미
있다면 α를 고려해 blend 하고, 아무것도 없다면 𝐼1를 그대로 그려준다.