# 20-07-06_29
# 임베딩과 LSTM

from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23]]

from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', value=0)   # Defalut (pre와 0 : 그만큼 이 조합이 많이 쓰인 다는 것)
print(pad_x)

word_size = len(token.word_index) + 1
print('전체 토큰 사이즈 :', word_size)               # 25 (전체 단어의 개수)

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM


### 모델
model = Sequential()
# model.add(Embedding(word_size, 10, input_length=5))   # 전체 단어의 수, 노드(layer's output)수, 인풋값(12, '5')
# model.add(Embedding(25, 10, input_length=5))          # (None, 5, 10) // 3차원
model.add(Embedding(25, 10))                          # x에 대한 인풋값을 명시하지 않았는데, 훈련이 된다. (= 임베딩과 LSTM을 동시에 사용할 시, 인풋을 명시안해도 된다)

model.add(LSTM(4))                                    # Conv1D 도 같은 방법으로 가능 (통상적으로 임베딩은 시계열이므로 LSTM을 많이 사용한다)
model.add(Dense(1, activation='sigmoid'))

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, None, 10)          250 : (25 * 10)
_________________________________________________________________
lstm_1 (LSTM)                (None, 3)                 168 : (4 * (10 + 1 + 3) * 3)
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 4
=================================================================
Total params: 422
Trainable params: 422
Non-trainable params: 0
_________________________________________________________________
LSTM layer param 계산식 = 4 * (input + bias + output) * output

'''


# ### 실행, 훈련, 평가
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# model.fit(pad_x, labels, epochs=30)

# acc = model.evaluate(pad_x, labels)[1]  # loss가 아닌 metrics 값을 빼겠다.
# print('\n acc : %.4f' % acc)