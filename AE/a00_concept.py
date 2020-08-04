# 20-08-04 Autoencoder

# 오토인코더의 개념

# PCA 차원축소 / feature importance 특성추출
# 오토인코더를 통해 나온 결과물은 x 결국 특징은

# 1. 특성 추출
# 2. GAN 과 연결?

'''
<Autoencoder>
어떤 데이터를 효율적으로 나타내기 위해서 고차원을 저차원으로 차원 축소하는 방법
= encode + decode

input(입력)rhk output(출력)이 동일하며 좌우를 대칭으로 구축된 구조

HL(hidden layer) 기준으로 IL(input layer) OL(output layer) 가 좌우 대칭
HL의 노드의 수는 몇 개로 구성하든지 상관없다
다만, 데이터의 차원축소 및 특징 추출이 목적인 경우는 IL의 노드의 수보다 작게한다.

HL의 node의 값을 code라 부른다
A = 데이터 X를 입력 받아 가중치의 곱과 합, 활성함수의 조합으로 HL의 node의 값을 계산한 상황 = encoder
B = code값에 적당한 w를 곱하고 활성함수를 적용해 OL에 도달하는데, 이때 OL은 IL의 값과 같아
원래 값으로 복원하는 과정 = Decoder
(A -> (code) -> B)

# a01_autoencoder.py 설명

IL(60000, 784) -> HL(32) -> OL(60000, 784)

mnist 기준
데이터 하나를 보았을 때 7이라는 값 외 배경들은 값이 0
그리고 7에 해당하는 부분은 1~255 값에서 전처리로 0~1의 값을 갖는다
차원을 축소하면서 0값을 가진 배경은 0~1값을 가진 것으로 압축되어
특성 추출느낌으로 압축이 되어진다

<sigmoid 에 관한 고찰>
- 앞 뒤가 똑같은 autoencoder 특성상 x 전처리 시, 0~1
- mse와 binary 는 결과치 보고 판단한다

'''