# 20-07-03_28
# vgg
# 전이학습?
# 이미지 쪽 이미지 분석할 때 가져다 쓸 수 있다.
# 서머리로 봐서 동일하게 유사하게 구성해도 된다.
# 얘가 이미지넷에서 준우승을 했기 때문에 그와 유사한 데이터셋등이 나오면 활용 가능하다.

from keras.applications import VGG16, VGG19
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from keras.optimizers import Adam

# 점심 시간에 다 다운 받아라
from keras.applications import Xception, ResNet101, ResNet101V2, ResNet152
from keras.applications import ResNet152V2, ResNet50, ResNet50V2, InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201, NASNetLarge, NASNetMobile
# model = VGG19()
# model = Xception()
# model =  ResNet101()
# model = ResNet101V2()
# model = ResNet152()
# model = ResNet152V2()
# model = ResNet50()
# model = ResNet50V2()
# model = InceptionV3()
# model = InceptionResNetV2()
# model = MobileNet()
# model = MobileNetV2()
# model = DenseNet121()
# model = DenseNet169()
# model = DenseNet201()
# model = NASNetLarge()
# model = NASNetMobile()

vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))   # (None, 224, 224, 3)
# vgg16.summary()

act = 'relu'
model = Sequential()

model.add(vgg16)
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation(act))
model.add(Dense(10, activation='softmax'))

model.summary()

# 잘만든 모델 가져다 쓰는 거 = 전이학습
# 이미지 모델에서 준우승 한 모델

# VGG16이랑 엮기
