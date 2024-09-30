"""
미리 사전에 훈련된 임베딩 벡터(pre-trained word embedding)들을 가져와 사용하는 방법
"""
"""
1. 사전 훈련된 임베딩을 사용하지 않는 경우
"""

import numpy as np
from collections import Counter
import gensim

# 긍, 부정을 판단하는 감성 분류 모델 예시. 긍정은 1 부정은 0
sentences = [
    "nice great best amazing",
    "stop lies",
    "pitiful nerd",
    "excellent work",
    "supreme quality",
    "bad",
    "highly respectable",
]
y_train = [1, 0, 0, 1, 1, 0, 1]

# 단어 토큰화를 수행
tokenized_sentences = [sent.split() for sent in sentences]
print("단어 토큰화 된 결과 :", tokenized_sentences)

# Counter() 모듈을 이용하여 각 단어의 등장 빈도수를 기록
word_list = []
for sent in tokenized_sentences:
    for word in sent:
        word_list.append(word)

word_counts = Counter(word_list)
print("총 단어수 :", len(word_counts))

# 등장 빈도순으로 정렬
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
print(vocab)

word_to_index = dict()
word_to_index["<PAD>"] = 0
word_to_index["<UNK>"] = 1

for index, word in enumerate(vocab):
    word_to_index[word] = index + 2

vocab_size = len(word_to_index)
print("패딩 토큰, UNK 토큰을 고려한 단어 집합의 크기 :", vocab_size)

print(word_to_index)


def texts_to_sequences(tokenized_X_data, word_to_index):
    encoded_X_data = []
    for sent in tokenized_X_data:
        index_sequences = []
        for word in sent:
            try:
                index_sequences.append(word_to_index[word])
            except KeyError:
                index_sequences.append(word_to_index["<UNK>"])
        encoded_X_data.append(index_sequences)
    return encoded_X_data


X_encoded = texts_to_sequences(tokenized_sentences, word_to_index)
print(X_encoded)

max_len = max(len(l) for l in X_encoded)
print("최대 길이 :", max_len)


def pad_sequences(sentences, max_len):
    features = np.zeros((len(sentences), max_len), dtype=int)
    for index, sentence in enumerate(sentences):
        if len(sentence) != 0:
            features[index, : len(sentence)] = np.array(sentence)[:max_len]
    return features


X_train = pad_sequences(X_encoded, max_len=max_len)
y_train = np.array(y_train)
print("패딩 결과 :")
print(X_train)

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


# 모델 설계
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(embedding_dim * max_len, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # embedded.shape == (배치 크기, 문장의 길이, 임베딩 벡터의 차원)
        embedded = self.embedding(x)

        # flattend.shape == (배치 크기, 문장의 길이 × 임베딩 벡터의 차원)
        flattened = self.flatten(embedded)

        # output.shape == (배치 크기, 1)
        output = self.fc(flattened)
        return self.sigmoid(output)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_dim = 100
simple_model = SimpleModel(vocab_size, embedding_dim).to(device)

criterion = nn.BCELoss()
optimizer = Adam(simple_model.parameters())

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32)
)
train_dataloader = DataLoader(train_dataset, batch_size=2)

for epoch in range(10):
    for inputs, targets in train_dataloader:
        # inputs.shape == (배치 크기, 문장 길이)
        # targets.shape == (배치 크기)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # outputs.shape == (배치 크기)
        outputs = simple_model(inputs).view(-1)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


"""
2. 사전 훈련된 임베딩을 사용하는 경우

사전 학습시킨 word2vec 모델 다운
!pip install gdown
!gdown https://drive.google.com/uc?id=1Av37IVBQAAntSe1X3MOAl5gvowQzd2_j
"""
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin.gz', binary=True)

embedding_matrix = np.zeros((vocab_size, 300))
print('임베딩 행렬의 크기 :', embedding_matrix.shape)


def get_vector(word):
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return None

# <PAD>를 위한 0번과 <UNK>를 위한 1번은 실제 단어가 아니므로 맵핑에서 제외
for word, i in word_to_index.items():
    if i > 2:
        temp = get_vector(word)
        if temp is not None:
            embedding_matrix[i] = temp

# <PAD>나 <UNK>의 경우는 사전 훈련된 임베딩이 들어가지 않아서 0벡터임
print(embedding_matrix[0])
print(word_to_index['great'])

# word2vec_model에서 'great'의 임베딩 벡터
# embedding_matrix[3]이 일치하는지 체크
print(np.all(word2vec_model['great'] == embedding_matrix[3]))


class PretrainedEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(PretrainedEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = True
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(embedding_dim * max_len, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        flattened = self.flatten(embedded)
        output = self.fc(flattened)
        return self.sigmoid(output)


pretraiend_embedding_model = PretrainedEmbeddingModel(vocab_size, 300).to(device)

criterion = nn.BCELoss()
optimizer = Adam(pretraiend_embedding_model.parameters())

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32))
train_dataloader = DataLoader(train_dataset, batch_size=2)

for epoch in range(10):
    for inputs, targets in train_dataloader:
        # inputs.shape == (배치 크기, 문장 길이)
        # targets.shape == (배치 크기)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # outputs.shape == (배치 크기)
        outputs = pretraiend_embedding_model(inputs).view(-1)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

