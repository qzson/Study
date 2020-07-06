# 20-07-06_29
# 임베디드

from keras.preprocessing.text import Tokenizer

text = '나는 맛있는 밥을 먹었다'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'나는': 1, '맛있는': 2, '밥을': 3, '먹었다': 4}

# 자연어 처리의 기본
# 임베디드 하면서 Tokenizer 하는 이유는?
# 자연어 처리를 하게 된다면, 단어 위주로 하던지 각 글자별로 할 수도 있다

x = token.texts_to_sequences([text])
print(x)    # [[1, 2, 3, 4]]
# 단지 인덱싱일 뿐 'x배'가 아니다

from keras.utils import to_categorical  # 원 핫 인코딩이 된다

word_size = len(token.word_index) + 1   # 쟤는 1부터 인덱스 시작이니, 0이없다 그래서 +1을 추가 해준 것
x = to_categorical(x, num_classes=word_size)
print(x)
# [[[0. 1. 0. 0. 0.]  
#   [0. 0. 1. 0. 0.]  
#   [0. 0. 0. 1. 0.]  
#   [0. 0. 0. 0. 1.]]]

# 데이터가 많아지면 엄청 커진다. 그런 문제점이 있다. 신문...
# 그렇다면 압축을 해야하는데 그래서 나온 것이 임베딩이다. (자연어처리와 시계열에서 많이 쓴다)