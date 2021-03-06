''' 200526 CNN 1200~ 1400~ 1500~ 1600~
 이미지와 관련된 뉴럴 네트워크 '''

''' 과제3 (완)
 summary 파라미터 개수 '''

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1)))       # (9, 9, 10)
 # 10, (2,2)          : 해당 레이어의 아웃풋, 2 x 2 로 자른다.
 # (10,10,1)          : (가로, 세로, 흑백) [1은 흑백 (1 or 3을 많이 보게 된다. 3은 칼라)]
 # (10000, 10, 10, 1) : 10000장, 이하 위와 같다. / (4차원 구성) / 행(10000) 무시하고 (input_shape=3차원 명시)
 # (z,a,b,c)          : 행(무시), 가로, 세로, 색 정도로 알고 있자.
 
''' Convolution layer 정확 설명
 첫번째 인자           : 필터의 수
 두번째 인자           : 커널의 (행, 열)
 이렇게도 쓰인다       : kernel_size=(2,2) = kernel_size=2 <같은 표현>
 
 padding              : 경계 처리 방법 정의
 - ‘valid’            : 유효한 영역만 출력. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작음.
 - ‘same’             : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일.

 input_shape          : 샘플 수를 제외한 입력 형태를 정의
                        (height, width, channel)=(행, 열, 채널수)로 정의. 흑백 1이고, 컬러(RGB) 3으로 설정.
 
 activation           : 활성화 함수 설정.
 - ‘linear’           : 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옴.
 - ‘relu’             : rectifier 함수, 은닉층에 주로 쓰임.
 - ‘sigmoid’          : 이진 분류 문제에서 출력층에 주로 쓰임.
 - ‘softmax’          : 다중 클래스 분류 문제에서 출력층에 주로 쓰임.
 https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/
 strides              : 기본값 = 1, 1일 경우 이미지가 겹쳐서 conv 된다. 값이 2,3...면 그 값만큼 큼직하게
                        conv 하게된다.(이미지가 클 경우 사용? 그 만큼 정확도는 떨어질 수 있다) '''

model.add(Conv2D(7, (3,3)))                               # (7, 7, 7)
 # 처음 자른 것을 가지고 또 한번 잘라서 특성을 찾는다. (cnn의 경우 몇 번씩 이 과정을 증폭하며 반복.)
model.add(Conv2D(5, (2,2), padding='same'))               # (7, 7, 5)
model.add(Conv2D(5, (2,2)))                               # (6, 6, 5)
 # model.add(Conv2D(5, (2,2), strides=2))                # (3, 3, 5) 두칸 두칸 두칸
# model.add(Conv2D(5, (2,2), strides=2))                  # (3, 3, 5)
# model.add(Conv2D(5, (2,2), strides=2, padding='same'))  # (3, 3, 5) => strides 가 우선순위다
 # 정확하지는 않아도 모델을 머리속에 그리면서 하는 것
model.add(MaxPooling2D(pool_size=2))                      # (3, 3, 5) 2D 이기 때문에, 2x2로
 # maxpooling 은 반드시 들어가야 하는 것은 아니다.
model.add(Flatten())                                      # 45 = (3 * 3 * 5)
 # conv 레이어의 끝은 항상 Flatten을 넣어야 한다.
model.add(Dense(1))
 # 서머리 상, 윗 레이어가 행무시하고 3차원인데, 안되는 경우다.
 # 데이터를 쫙 펴야한다.

model.summary()
''' summary
 _________________________________________________________________
 Layer (type)                 Output Shape              Param #   
 =================================================================
 conv2d_1 (Conv2D)            (None, 10, 10, 10)        50        
 _________________________________________________________________
 conv2d_2 (Conv2D)            (None, 10, 10, 7)         287       
 _________________________________________________________________
 conv2d_3 (Conv2D)            (None, 10, 10, 5)         145       
 =================================================================
 Total params: 482
 Trainable params: 482
 Non-trainable params: 0
 _________________________________________________________________
 : padding 다 지웟을 때 서머리 ____________________________________
 Layer (type)                 Output Shape              Param #
 =================================================================
 conv2d_1 (Conv2D)            (None, 9, 9, 10)          50
 _________________________________________________________________
 conv2d_2 (Conv2D)            (None, 8, 8, 7)           287
 _________________________________________________________________
 conv2d_3 (Conv2D)            (None, 7, 7, 5)           145
 =================================================================
 
 10 10 10 에서 9 8 7 이 되는 이유 = 손 코딩, 그려서 확인
 
 파라미터 개수 계산 공식

 1. [i * (f * f) * o] + o (이 방법은 보류)
 i : input
 f : kernal
 o : output 

 2. cnn param = (input * karnal * kernal + bias) output
 3. dense param = (input + bias) * output '''

''' Maxpooling 기능
 이미지 중에서 가장 큰 값만 추출해내는 것 '''

# 여기까지가 cnn 하이퍼 파라미터 튜닝의 범위
# 하이퍼 파라미터 튜닝을 이것저것 하면서 실험해보자
# maxpooling 은 3번씩도 가능하다.

# cnn 모델링 데이터 넣어서 작업할 것
# x 값이 100만개 gpu 버전으로 바꿀 것
# conv 레이어는 정제된 데이터를 줬지만, 가장 많이 쓰는 예제들의 데이터를 들어간다