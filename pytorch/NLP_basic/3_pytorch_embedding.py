"""
파이토치에서는 임베딩 벡터를 사용하는 방법이 크게 두 가지가 있음.
1. 임베딩 층(embedding layer)을 만들어 훈련 데이터로부터 처음부터 임베딩 벡터를 학습하는 방법
2. 미리 사전에 훈련된 임베딩 벡터(pre-trained word embedding)들을 가져와 사용하는 방법
"""

"""
임베딩 층(embedding layer)을 만들어 훈련 데이터로부터 처음부터 임베딩 벡터를 학습

임베딩 층의 입력으로 사용하기 위해서 입력 시퀀스의 각 단어들은 모두 정수 인코딩이 되어있어야 함
어떤 단어 → 단어에 부여된 고유한 정수값 → 임베딩 층 통과 → 밀집 벡터

"""
import torch
import torch.nn as nn

# 임의의 문장으로부터 단어 집합을 만들고 각 단어에 정수를 부여
train_data = "you need to know how to code"

# 중복을 제거한 단어들의 집합인 단어 집합 생성.
word_set = set(train_data.split())

# 단어 집합의 각 단어에 고유한 정수 맵핑.
vocab = {word: i + 2 for i, word in enumerate(word_set)}
vocab["<unk>"] = 0  # 모르는 단어는 0으로
vocab["<pad>"] = 1  # 패딩 토큰은 1로
print(vocab)

# 단어 집합의 크기만큼의 행을 가지는 테이블 생성. 임베딩 벡터의 차원은 3으로 설정
embedding_table = torch.FloatTensor(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.2, 0.9, 0.3],
        [0.1, 0.5, 0.7],
        [0.2, 0.1, 0.8],
        [0.4, 0.1, 0.1],
        [0.1, 0.8, 0.9],
        [0.6, 0.1, 0.1],
    ]
)

# 임의의 문장 'you need to run'에 대해서 룩업 테이블을 통해 임베딩 벡터들을 가져오기
sample = 'you need to run'.split()
idxes = []

# 각 단어를 정수로 변환
for word in sample:
    try:
        idxes.append(vocab[word])
    # 단어 집합에 없는 단어일 경우 <unk>로 대체된다.
    except KeyError:
        idxes.append(vocab['<unk>'])

idxes = torch.LongTensor(idxes)


# 각 정수를 인덱스로 임베딩 테이블에서 값을 가져온다.
lookup_result = embedding_table[idxes, :]
print(lookup_result)

# 임베딩 층 사용하기
train_data = 'you need to know how to code'

# 중복을 제거한 단어들의 집합인 단어 집합 생성.
word_set = set(train_data.split())

# 단어 집합의 각 단어에 고유한 정수 맵핑.
vocab = {tkn: i+2 for i, tkn in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

"""
nn.Embedding()을 사용하여 학습가능한 임베딩 테이블을 만들기

num_embeddings : 임베딩을 할 단어들의 개수. 다시 말해 단어 집합의 크기
embedding_dim : 임베딩 할 벡터의 차원. 사용자가 정해주는 하이퍼파라미터
padding_idx : 선택적으로 사용하는 인자. 패딩을 위한 토큰의 인덱스를 알려줌
"""
embedding_layer = nn.Embedding(num_embeddings=len(vocab),
                               embedding_dim=3,
                               padding_idx=1)

print(embedding_layer.weight)
