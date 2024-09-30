"""
원-핫 인코딩은 희소 표현의 문제가 있음
* 희소표현: 벡터 또는 행렬(matrix)의 값이 대부분이 0으로 표현되는 방법
ex) 강아지 = [ 0 0 0 0 1 0 0 0 0 0 0 0 ... 중략 ... 0] # 이 때 1 뒤의 0의 수는 9995개. 차원은 10,000

밀집 표현: 사용자가 설정한 값으로 모든 단어의 벡터 표현의 차원을 맞춤
Ex) 강아지 = [0.2 1.8 1.1 -2.1 1.1 2.8 ... 중략 ...] # 이 벡터의 차원은 128

단어를 밀집 벡터(dense vector)의 형태로 표현하는 방법을 워드 임베딩(word embedding)
워드 임베딩 과정을 통해 나온 결과라고 하여 임베딩 벡터(embedding vector)으로 표현
"""

"""
워드투벡터
"""
import re
import urllib.request
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize

# 데이터 다운로드
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml",
    filename="ted_en-20160408.xml",
)

targetXML = open("ted_en-20160408.xml", "r", encoding="UTF8")
target_text = etree.parse(targetXML)

# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
parse_text = "\n".join(target_text.xpath("//content/text()"))

# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# 해당 코드는 괄호로 구성된 내용을 제거.
content_text = re.sub(r"\([^)]*\)", "", parse_text)

# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
sent_text = sent_tokenize(content_text)

# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
normalized_text = []
for string in sent_text:
    tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
    normalized_text.append(tokens)

# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
result = [word_tokenize(sentence) for sentence in normalized_text]

print("총 샘플의 개수 : {}".format(len(result)))

# 샘플 3개만 출력
for line in result[:3]:
    print(line)

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

model = Word2Vec(
    sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0
)

model_result = model.wv.most_similar("man")
print(model_result)

model.wv.save_word2vec_format("eng_w2v")  # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format("eng_w2v")  # 모델 로드

#  man과 유사한 단어를 출력
model_result = loaded_model.most_similar("man")
print(model_result)
