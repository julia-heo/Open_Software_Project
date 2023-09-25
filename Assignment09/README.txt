실행하는 방법:
	파이썬 설치
	터미널에서 skeleton code 있는 폴더로 이동 후
		pip install jupyter
		pip install -r requirements.txt
		jupyter notebook

1. two_layer_net.ipynb
forward pass로 score과 loss를 계산한다.
그 결과를 이용해 W1, W2, b1, b2에 대해 gradient를 계산한다.
gradient를 이용해 W1, W2, b1, b2를 업데이트 하며 training을 진행한다.
결과를 보고, 값을 바꿔 개선한다.

2.neural_net.py
1번 코드 진행에 중심적인 함수들을 제공한다.
	1) loss
		y가 없다면 score을, 있다면 score과 loss, gradient를 반환한다.
	2) train
		W1, W2, b1, b2를 업데이트 하며 training을 진행한다.
	3) predict
		각 클래스의 score을 계산하고 가장 높은 점수를 가진 클래스를 반환한다.