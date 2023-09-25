import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
#from vgg16_full import *
from resnet50 import *

# Set device : 디바이스 세팅_ cpu

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 디바이스가 GPU & CUDA를 가지고 있는지 확인. 없으면 cpu로
device = torch.device('cpu')

# Image Preprocessing 이미지 전처리
# Data augmentation and preprocessing using transform function: transform func을 이용한 데이터 확대 및 전처리
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10 Dataset을 가져온다
train_dataset = torchvision.datasets.CIFAR10(root='../osproj/data/',
                                             train=True,
                                             transform=transform_train,
                                             download=False) # Change Download-flag "True" at the first excution.
                                            # download: 처음 사용하는 경우 True

test_dataset = torchvision.datasets.CIFAR10(root='../osproj/data/',
                                            train=False,
                                            transform=transform_test)


# data loader: DataLoader()를 이용해 데이터를 load
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,          # 한 번의 iteration에 사용된 교육 예제의 수
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)
###########################################################
# Choose model
model = ResNet50_layer4().to(device)
PATH = './resnet50_epoch285.ckpt' # test acc would be almost 80

# vgg16 사용
# model = vgg16().to(device)
# PATH = './vgg16_epoch250.ckpt'  # test acc would be almost 85
##############################################################
checkpoint = torch.load(PATH)   # ① : 체크포인트를 저장한 경우 모델의 매개변수만 포함  ② : 저장된 체크포인트 파일에 변수 로드
model.load_state_dict(checkpoint)

# Train Model
# Hyper-parameters
num_epochs = 1  # students should train 1 epoch because they will use cpu
learning_rate = 0.001

# Loss and optimizer 고르기
criterion = nn.CrossEntropyLoss() # Loss: 실측값과 예측 출력 사이의 차이를 가져온다 # Cross Entropy Loss: LogSoftmax(), NLLLoss() 포함
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)  # train_loader: 데이터 로드했던 변수
current_lr = learning_rate

for epoch in range(num_epochs):

    model.train()       # Set model to train mode
    train_loss = 0

    for batch_index, (images, labels) in enumerate(train_loader):       # Get both index and data using enumerate()

        # print(images.shape)
        images = images.to(device)  # "images" = "inputs"
        labels = labels.to(device)  # "labels" = "targets"

        # Forward pass
        # modle에 입력하여 예측 output을 얻고 criterion()에 따라 손실을 계산
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize

        optimizer.zero_grad()       # gradients를 0으로 설정_ PyTorch가 gradients를 누적하기 때문
        loss.backward()             # loss.backward()으로 backpropagation 수행
        optimizer.step()

        train_loss += loss.item()

        if (batch_index + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, batch_index + 1, total_step, train_loss / (batch_index + 1)))

    # Decay learning rate
    if (epoch + 1) % 20 == 0:           # optimal point에 도달하기 위해 학습 속도를 줄인다
        current_lr /= 3
        update_lr(optimizer, current_lr)
        torch.save(model.state_dict(), './resnet50_epoch' + str(epoch+1)+'.ckpt')

# Save the model checkpoint: 모델을 save
torch.save(model.state_dict(), './resnet50_final.ckpt')     # obj : saved object , f: 파일 이름을 포함하는 string
# ① : 모델의 매개변수만 저장  ② : 체크포인트 파일을 다른 변수와 함께 저장

model.eval()                                                # model을 평가모드로 설정
with torch.no_grad():                                       # Autograd 엔진에 영향을 주고 비활성화. 메모리 사용량을 줄이고 계산 속도를 높이지만 backprop은 할 수 없다
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        # test에선 backpropagate loss하지 않음
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
