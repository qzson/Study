# 20-07-06_29
# 임베디드

'''
컴퓨터는 텍스트를 이해할 수 없기 때문에 텍스트를 정제하는 전처리가 꼭 필요
- 텍스트를 잘게 나누는 것 (작게 나누어진 하나의 단위 : token)
- 그 과정을 tokenization라고 한다
'''

from keras.preprocessing.text import Tokenizer # 단어의 빈도 수를 쉽게 계산해주는 함수

text = '나는 맛있는 밥을 먹었다'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)     # 각 단어에 매겨진 인덱스 값 출력
# {'나는': 1, '맛있는': 2, '밥을': 3, '먹었다': 4}

print(token.word_counts)    # 단어의 빈도 수를 계산한 결과 출력
# OrderedDict([('나는', 1), ('맛있는', 1), ('밥을', 1), ('먹었다', 1)])

print(token.document_count) # 몇 개의 문장이 있는지
# 1

print(token.word_docs)      # 각 단어들이 몇 개의 문장에서 나오는지
# defaultdict(<class 'int'>, {'먹었다': 1, '나는': 1, '밥을': 1, '맛있는': 1})

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

# 원-핫 인코딩을 그대로 사용하면 벡터의 길이가 너무 길어진다는 단점이 있다.
# 예로, 1만 개의 단어 토큰으로 이루어진 말뭉치를 다룰 때, 이 데이터를 원-핫 인코딩으로 벡터화하면,
# 9999개의 0과 하나의 1로 이루어진 단어 벡터를 1만 개나 만들어야 한다.
# 이러한 공간적 낭비를 해결하기 위해 등장한 것이 단어 임베딩이다. (자연어처리와 시계열에서 많이 쓴다)

# 임베딩은 주어진 배열을 정해진 길이로 압축시킨다.