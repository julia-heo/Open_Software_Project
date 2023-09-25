실행하는 방법:
	window에서 실행
	파이썬 설치
	anaconda 설치
	Anaconda Prompt 접속)
		conda create –n PyTorch_envpython=3.6 pytorch torchvision –c pytorch -y
		conda activate PyTorch_envpython
		pip install opencv-python
		pip install torchvision
	Pycharm 설치
		new project 생성
		생성한 가상환경 PyTorch_envpython를 선택
		코드 디버깅 후 실행
	- UNet 실행 시 main.py:
		model = Unet(3, 22)
		PATH = 'UNet_trained_model.pth'
	- resnet을 이용한 Unet 실행 시 main.py:
		model = UNetWithResnet50Encoder(22)
		PATH = 'resnet_encoder_unet.pth'

1. UNet_skeleton.py
	UNet을 구현하기 위한 코드이다. UNet동작에 맞게 channel size를 지정하고, conv up과 conv down을 진행한다.

2. resnet_encoder_unet_skeleton.py 
	resNet을 이용해 UNet을 구현하는 코드이다. assignment10의 resnet코드와 일치하는 부분이 존재하고, 이를 이용해 Unet을 형성한다.

3. modules_skeleton.py
	training을 진행하기 위한 코드이다. train_model, get_loss_train함수 등이 정의되어 있다.
4. main_skeleton.py
	프로그램 실행을 위한 main 코드이다. 1번과 2번 중 모델을 선택하고 train을 진행한 뒤, 결과에 따른 이미지를 생성한다.
