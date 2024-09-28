import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
"""
훈련 데이터 전처리하기
"""
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))  # 중복을 제거한 문자 집합 생성
char_dic = {c: i for i, c in enumerate(char_set)}  # 각 문자에 정수 인코딩
print(char_dic)

dic_size = len(char_dic)
print('문자 집합의 크기 : {}'.format(dic_size))

# 하이퍼파라미터 설정
hidden_size = dic_size  # hidden_size(은닉 상태의 크기)를 입력의 크기와 동일하게 줬는데, 이는 사용자의 선택으로 다른 값을 줘도 무방
sequence_length = 10  # 임의 숫자 지정
learning_rate = 0.1

# 데이터 구성
x_data = []
y_data = []

# 임의로 지정한 sequence_length 값인 10의 단위로 샘플들을 잘라서 데이터를 만듬
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x_data.append([char_dic[c] for c in x_str])  # x str to index
    y_data.append([char_dic[c] for c in y_str])  # y str to index

print(x_data[0])  # if you wan에 해당됨.
print(y_data[0])  # f you want에 해당됨.

# x_one_hot = [np.eye(dic_size)[x] for x in x_data]  # x 데이터는 원-핫 인코딩
x_one_hot = torch.nn.functional.one_hot(torch.tensor(x_data), num_classes=dic_size).float()
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)

print('훈련 데이터의 크기 : {}'.format(X.shape))
print('레이블의 크기 : {}'.format(Y.shape))

print(X[0])
print(Y[0])

"""
모델 구현하기
"""
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):  # 현재 hidden_size는 dic_size와 같음.
        super(Net, self).__init__()
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x


net = Net(dic_size, hidden_size, 2)  # 이번에는 층을 두 개 쌓습니다.

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), learning_rate)

for i in range(100):
    optimizer.zero_grad()
    outputs = net(X)  # (170, 10, 25) 크기를 가진 텐서를 매 에포크마다 모델의 입력으로 사용
    loss = criterion(outputs.view(-1, dic_size), Y.view(-1))  # view를 사용해 2차원 텐서로 변환. - 정확도 측정을 위해 변환
    loss.backward()
    optimizer.step()

    # results의 텐서 크기는 (170, 10)
    results = outputs.argmax(dim=2)
    predict_str = ""
    for j, result in enumerate(results):
        if j == 0:  # 처음에는 예측 결과를 전부 가져오지만
            predict_str += ''.join([char_set[t] for t in result])
        else:  # 그 다음에는 마지막 글자만 반복 추가
            predict_str += char_set[result[-1]]

    print(predict_str)
