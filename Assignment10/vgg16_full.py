import torch.nn as nn
import math

###### VGG16 #############
class VGG(nn.Module):
    def __init__(self, features):                   # VGG 객체가 생성되면 동작
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(            # 고정된 부분을 class 내에 설계
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():                     # 모델 클래스에서 정의된 layer들을 iterable로 차례로 반환
            if isinstance(m, nn.Conv2d):             # isinstance: 차례로 layer을 입력하여, layer의 형태를 반환
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):                   # Network forward에서 호출된다. forward prop을 진행시키는 함수
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):                 # make_layer가 vgg16에게 kernelSize정보 받아 Sequential만들어 VGG에게 전달
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':                                                                    # max pooling의 경우
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]                           # kernel size 2, stride 2fh Max Pooling 진행
        else:                                                                           # Convolution+ReLU의 경우
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)                # conv 계산
            if batch_norm:                                                              # Batch Normalization이 있는 경우
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]            # Batch Normalization, ReLU 진행
            else:                                                                       # Batch Normalization이 없는 경우
                layers += [conv2d, nn.ReLU(inplace=True)]                               # ReLU 진행
            in_channels = v                                                             # 다음 filter size를 위해 in_channels에 값 저장
    return nn.Sequential(*layers)                                                       # sequential container 반환

def vgg16():                        # main.py함수에서 호출
    # cfg shows 'kernel size'
    # 'M' means 'max pooling', else means 'Convolution+ReLU'
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return VGG(make_layers(cfg))   # VGG 객체 반환