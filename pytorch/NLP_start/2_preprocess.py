"""
토큰화
"""
# spaCy 사용하기
import spacy

en_text = "A Dog Run back corner near spare bedrooms"

spacy_en = spacy.load('en_core_web_sm')


def tokenize(en_text):
    return [tok.text for tok in spacy_en.tokenizer(en_text)]


print(tokenize(en_text))

# nltk 사용하기
import nltk
from nltk.tokenize import word_tokenize

# nltk.download('punkt')
print(word_tokenize(en_text))

# 띄어쓰기로 토큰화
print(en_text.split())

# 한국어 띄어쓰기 토큰화
kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"

print(kor_text.split())

# mecab 사용

# 문자 토큰화
print(list(en_text))

"""
단어 집합(Vocabulary) 생성
"""
import urllib.request
import pandas as pd
from nltk import FreqDist
from konlpy.tag import Okt
import numpy as np
import matplotlib.pyplot as plt

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
data = pd.read_table('ratings.txt')  # 데이터프레임에 저장

print(data[:10])

sample_data = data[:100]  # 임의로 100개만 저장

# 정규 표현식 사용하여 한글과 공백을 제외하고 모두 제거
sample_data['document'] = sample_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)

# 불용어 정의
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

tokenizer = Okt()
tokenized = []
for sentence in sample_data['document']:
    temp = tokenizer.morphs(sentence)  # 토큰화
    temp = [word for word in temp if not word in stopwords]  # 불용어 제거
    tokenized.append(temp)

print(tokenized[:10])

# 단어 집합 생성. NLTK에서는 빈도수 계산 도구인 FreqDist()를 지원
vocab = FreqDist(np.hstack(tokenized))
print('단어 집합의 크기 : {}'.format(len(vocab)))

# 단어를 키(key)로, 단어에 대한 빈도수가 값(value)으로 저장되어져 있습니다. vocab에 단어를 입력하면 빈도수를 리턴
print(vocab['재밌'])

vocab_size = 500

# 상위 vocab_size개의 단어만 보존
vocab = vocab.most_common(vocab_size)
print('단어 집합의 크기 : {}'.format(len(vocab)))

"""
3. 각 단어에 고유한 정수 부여
"""
# enumerate(): 순서가 있는 자료형(list, set, tuple, dictionary, string)을 입력으로 받아 인덱스를 순차적으로 함께 리턴
word_to_index = {word[0]: index + 2 for index, word in enumerate(vocab)}
word_to_index['pad'] = 1
word_to_index['unk'] = 0

# 기존의 훈련 데이터에서 각 단어를 고유한 정수로 부여하는 작업
encoded = []
for line in tokenized: #입력 데이터에서 1줄씩 문장을 읽음
    temp = []
    for w in line: #각 줄에서 1개씩 글자를 읽음
        try:
            temp.append(word_to_index[w])  # 글자를 해당되는 정수로 변환
        except KeyError:  # 단어 집합에 없는 단어일 경우 unk로 대체된다.
            temp.append(word_to_index['unk'])  # unk의 인덱스로 변환

    encoded.append(temp)

print(encoded[:10])

"""
길이가 다른 문장들을 모두 동일한 길이로 바꿔주는 패딩(padding)
"""
# 길이가 다른 리뷰들을 모두 동일한 길이로 바꿔주는 패딩 작업을 진행
max_len = max(len(l) for l in encoded)
print('리뷰의 최대 길이 : %d' % max_len)
print('리뷰의 최소 길이 : %d' % min(len(l) for l in encoded))
print('리뷰의 평균 길이 : %f' % (sum(map(len, encoded))/len(encoded)))
plt.hist([len(s) for s in encoded], bins=50)
plt.xlabel('length of sample')
plt.ylabel('number of sample')
plt.show()

# 가장 길이가 긴 리뷰의 길이는 62. 모든 리뷰의 길이를 62로 통일
for line in encoded:
    if len(line) < max_len: # 현재 샘플이 정해준 길이보다 짧으면
        line += [word_to_index['pad']] * (max_len - len(line))  # 나머지는 전부 'pad' 토큰으로 채운다.

print('리뷰의 최대 길이 : %d' % max(len(l) for l in encoded))
print('리뷰의 최소 길이 : %d' % min(len(l) for l in encoded))
print('리뷰의 평균 길이 : %f' % (sum(map(len, encoded))/len(encoded)))

print(encoded[:3])


