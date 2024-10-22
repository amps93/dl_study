# Subword Tokenizer

## OOV (out-of-vocabulary)

기계가 모르는 단어가 등장하면 그 단어를 단어 집합에 없는 단어란 의미에서 OOV 또는 UNK라고 표현

## 서브워드 분리

하나의 단어는 더 작은 단위의 의미있는 여러 서브워드들 (ex- birthplace = birth + place) 조합으로 구성된 경우가 많음

하나의 단어를 여러 서브워드로 분리해서 단어를 인코딩 및 임베딩하겠다는 의도를 가진 전처리 작업
