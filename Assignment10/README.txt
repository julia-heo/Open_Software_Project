실행하는 방법:
	window에서 실행
	파이썬 설치
	anaconda 설치
	Anaconda Prompt 접속)
		conda create –n PyTorch_envpython=3.6 pytorch torchvision –c pytorch -y
		conda activate PyTorch_envpython
	Pycharm 설치
		new project 생성
		생성한 가상환경 PyTorch_envpython를 선택
		코드 디버깅 후 실행
	- vgg 실행 시 main.py:
		from vgg16_full import *
		model = vgg16().to(device)
		PATH = './vgg16_epoch250.ckpt' 
	- resnet 실행 시 main.py:
		from resnet50 import * 
		model = ResNet50_layer4().to(device)
		PATH = './resnet50_epoch285.ckpt

1. main.py
	2, 3번의 모델을 이용해 cnn을 구현할 코드이다.

2. vgg16_fully.py
	main.py에서 model를 vgg16()로 설정하여 vgg16_fully.py의 객체와 함수를 이용한다.

3. resnet50.py
	main.py에서 model를 ResNet50_layer4()로 설정하여 resnet50.py의 객체와 함수를 이용한다. ResNet을 구현한 코드이다. 
