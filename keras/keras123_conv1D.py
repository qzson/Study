# 20-07-06_29
# 임베딩 제외하고 conv1d 생성


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
print(pad_x.shape)      # (12, 5)
pad_x = pad_x.reshape(-1, 5, 1)
print(pad_x.shape)      # (12, 5, 1)

word_size = len(token.word_index) + 1
print('전체 토큰 사이즈 :', word_size)               # 25 (전체 단어의 개수)


### 모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM, Conv1D

model = Sequential()
model.add(Conv1D(word_size, 2, input_shape=(5, 1)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()


### 실행, 훈련, 평가
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]  # loss가 아닌 metrics 값을 빼겠다.
print('\n acc : %.4f' % acc)