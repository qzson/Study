# 20-07-06_29
# 임베디드

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
# <너무 2번, 참 2번 일 때,>
# {'너무': 1, '참': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, ' 10, '한번': 11, '더': 12, '보고': 13, '싶네요': 14, '글쎄요': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 9  해 
# 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23}

# <너무 2번, 참 3번 일 때,> : 단어가 많은 것이 맨 앞 인덱스로 정렬
# {'참': 1, '너무': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한번': 11, '더': 12, '보고': 13, '싶네요': 14, '글쎄요': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 
# 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23}

print(token.word_counts)
# OrderedDict([('너무', 2), ('재밌어요', 1), ('참', 3), ('최고에요', 1), ('잘', 1), ('만든', 1), ('영화 
# 에요', 1), ('추천하고', 1), ('싶은', 1), ('영화입니다', 1), ('한', 1), ('번', 1), ('더', 1), ('보고', 
# 1), ('싶네요', 1), ('글쎄요', 1), ('별로에요', 1), ('생각보다', 1), ('지루해요', 1), ('연기가', 1), 
# ('어색해요', 1), ('재미없어요', 1), ('재미없다', 1), ('재밌네요', 1)])
print(token.document_count)
# 12

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23]]

from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', value=0)   # Defalut (pre와 0 : 그만큼 이 조합이 많이 쓰인 다는 것)
print(pad_x)
'''
pad_sequence '와꾸 맞춰주기'
# padding = 'post' (0 이 뒤로 채워진다)
(2,) : [3, 7],0,0,0
(1,) : [2] ,0,0,0,0
(3,) : [4,5,11],0,0
(5,) : [5,4,3,2,6 ]
>> (4,5)

# padding = 'pre' (시계열에 적합. 그러나 다른 것도 보통 이렇게 한다)
(2,) : 0,0,0,[3, 7]
(1,) : 0,0,0,0, [2]
(3,) : 0,0,[4,5,11]
(5,) : [5,4,3,2,6 ]
>> (4,5)
'''

word_size = len(token.word_index) + 1
print('전체 토큰 사이즈 :', word_size)               # 25 (전체 단어의 개수)

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten


### 모델
model = Sequential()
model.add(Embedding(word_size, 10, input_length=5))
# model.add(Embedding(25, 10, input_length=5))          # (None, 5, 10)

# 전체 단어 크기를 넣는 것이 가장 좋은데, 다른 수를 넣어도 모델이 돌아간다 (해당 값에 맞춰서 벡터값을 설정해준다..?)
# param {1:전체 단어의 수, 2:노드(layer's output)수, 3:인풋값(12, '5')}
# = 1:입력될 단어의 수, 2:임베딩 후 출력되는 벡터 크기 10, 3:단어의 수만큼이 아니라 매 번 5개 씩만 넣겠다

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()


### 실행, 훈련, 평가
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]  # loss가 아닌 metrics 값을 빼겠다.
print('\n acc : %.4f' % acc)