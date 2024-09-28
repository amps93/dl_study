from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

print('단어 토큰화1 :', word_tokenize(
    "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

print('단어 토큰화2 :', WordPunctTokenizer().tokenize(
    "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

print('단어 토큰화3 :', text_to_word_sequence(
    "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

"""
표준으로 쓰이고 있는 토큰화 방법 중 하나인 Penn Treebank Tokenization의 규칙에 대해서 소개하고, 토큰화의 결과를 확인
"""
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print('트리뱅크 워드토크나이저 :',tokenizer.tokenize(text))

"""
문장 토큰화
"""
from nltk.tokenize import sent_tokenize

text = ("His barber kept his word. But keeping such a huge secret to himself was driving him crazy. "
        "Finally, the barber went up a mountain and almost to the edge of a cliff. "
        "He dug a hole in the midst of some reeds. He looked about, to make sure no one was near.")
print('문장 토큰화1 :', sent_tokenize(text))

text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
print('문장 토큰화2 :', sent_tokenize(text))  # NLTK는 단순히 마침표를 구분자로 하여 문장을 구분하지 않았기 때문에, Ph.D.를 문장 내의 단어로 인식하여 성공적으로 인식

"""
한국어 토큰화: kss 사용
pipenv install kss
"""
import kss

text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
print('한국어 문장 토큰화 :', kss.split_sentences(text))

from konlpy.tag import Okt
from konlpy.tag import Kkma

okt = Okt()
kkma = Kkma()

print('OKT 형태소 분석 :', okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 품사 태깅 :', okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 명사 추출 :', okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))