"""
원-핫 인코딩
단어 집합의 크기를 벡터의 차원으로 하고,
표현하고 싶은 단어의 인덱스에 1의 값을 부여하고,
다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식
"""
from konlpy.tag import Okt
okt = Okt()
token = okt.morphs("나는 자연어 처리를 배운다")
print(token)

word2index = {}
for voca in token:
     if voca not in word2index.keys():
       word2index[voca] = len(word2index)
print(word2index)


def one_hot_encoding(word, word2index):
    one_hot_vector = [0] * (len(word2index))
    index = word2index[word]
    one_hot_vector[index] = 1
    return one_hot_vector


print(one_hot_encoding("자연어", word2index))

