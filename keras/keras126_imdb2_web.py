# 20-07-06_29
# self 실습

from keras.datasets import imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


### 1. data
(x_train, y_train), (x_test, y_test) = imdb.load_data()
# 영화 리뷰는 X_train에, 감성 정보는 y_train에 저장된다.
# 테스트용 리뷰는 X_test에, 테스트용 리뷰의 감성 정보는 y_test에 저장된다.

print('train용 리뷰 개수 : {}'.format(len(x_train))) # train용 리뷰 개수 : 25000
print('test용 리뷰 개수 : {}'.format(len(x_test)))   # test용 리뷰 개수 : 25000
num_classes = max(y_train) + 1
print('카테고리 : {}'.format(num_classes))           # 카테고리 : 2
# y_train는 0부터 시작해서 레이블을 부여, y_train에 들어 있는 가장 큰 수에 +1을 하여 출력 = 카테고리가 총 몇 개

print(x_train[0])
print(y_train[0])
# X_train[0]에는 숫자들 존재. 이 데이터는 토큰화와 정수 인코딩이라는 텍스트 전처리가 끝난 상태
# '1' 긍정의 1의 값

### 25,000개의 훈련용 리뷰의 길이 분포 그래프 시각화
len_result = [len(s) for s in x_train]

print('리뷰의 최대 길이 : {}'.format(np.max(len_result)))
print('리뷰의 평균 길이 : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
# plt.show()
# 리뷰의 최대 길이 : 2494
# 리뷰의 평균 길이 : 238.71364
# >> 대체적으로 1,000이하의 길이 // 특히 100~500길이를 가진 데이터가 많은 것을 확인
# >> 반면, 가장 긴 길이를 가진 데이터는 길이가 2,000이 넘는 것도 확인

### 레이블의 분포
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("각 레이블에 대한 빈도수:")
print(np.asarray((unique_elements, counts_elements)))
# 각 레이블에 대한 빈도수:
# [[    0     1]
#  [12500 12500]]
# 두 레이블 0과 1은 각각 12,500개로 균등한 분포

## x_train의 숫자가 어떤 단어인지 확인 <index_to_word에 인덱스를 넣으면 전처리 전에 어떤 단어였는지 확인 가능>
word_to_index = imdb.get_word_index()
index_to_word={}
for key, value in word_to_index.items():
    index_to_word[value] = key

print('빈도수 상위 1번 단어 : {}'.format(index_to_word[1]))         # 빈도수 상위 1번 단어 : the
print('빈도수 상위 3941번 단어 : {}'.format(index_to_word[3941]))   # 빈도수 상위 3941번 단어 : journalist

## x_train[0]이 인덱스로 바뀌기 전, 어떤 단어들이었는지 확인 (인덱스로 바꾸기 전에도 어느정도 전처리 진행된 상태라 완벽 X)
for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
      index_to_word[index]=token
print(' '.join([index_to_word[index] for index in x_train[0]]))
