"""
문서 단어 행렬(Document-Term Matrix, DTM): 다수의 문서에서 등장하는 각 단어들의 빈도를 행렬로 표현한 것

문서 단어 행렬(Document-Term Matrix)의 한계

1) 희소 표현(Sparse representation)
원-핫 벡터나 DTM과 같은 대부분의 값이 0인 표현을 희소 벡터(sparse vector) 또는 희소 행렬(sparse matrix)라고 부르는데,
희소 벡터는 많은 양의 저장 공간과 높은 계산 복잡도를 요구

2) 단순 빈도 수 기반 접근
"""

"""
TF-IDF(단어 빈도-역 문서 빈도, Term Frequency-Inverse Document Frequency)
"""
import pandas as pd  # 데이터프레임 사용을 위해
from math import log  # IDF 계산을 위해

docs = [
    "먹고 싶은 사과",
    "먹고 싶은 바나나",
    "길고 노란 바나나 바나나",
    "저는 과일이 좋아요",
]
vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()

# 총 문서의 수
N = len(docs)


def tf(t, d):
    return d.count(t)


def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N / (df + 1))


def tfidf(t, d):
    return tf(t, d) * idf(t)


result = []

# 각 문서에 대해서 아래 연산을 반복
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns=vocab)
print(tf_)

result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index=vocab, columns=["IDF"])
print(idf_)

result = []
for i in range(N):
  result.append([])
  d = docs[i]
  for j in range(len(vocab)):
    t = vocab[j]
    result[-1].append(tfidf(t, d))

tfidf_ = pd.DataFrame(result, columns=vocab)
print(tfidf_)

"""
사이킷런 사용
"""
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',
]

vector = CountVectorizer()

# 코퍼스로부터 각 단어의 빈도수를 기록
print(vector.fit_transform(corpus).toarray())

# 각 단어와 맵핑된 인덱스 출력
print(vector.vocabulary_)

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',
]

tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)
