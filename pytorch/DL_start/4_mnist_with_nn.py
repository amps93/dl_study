import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

# 레이블 데이터 타입을 정수형으로 변환
mnist.target = mnist.target.astype(np.int8)
print(len(mnist.target))

X = mnist.data / 255  # 0-255값을 [0,1] 구간으로 정규화
y = mnist.target

plt.imshow(X[0].reshape(28, 28), cmap='gray')
plt.show()
print("이 이미지 데이터의 레이블은 {:.0f}이다".format(y[0]))

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=0)

# 텐서로 변환
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# TensorDataset 객체 생성
ds_train = TensorDataset(X_train, y_train)
ds_test = TensorDataset(X_test, y_test)

# DataLoader 객체 생성
loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)

# 모델 생성
model = nn.Sequential()
model.add_module('fc1', nn.Linear(28*28*1, 100))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100, 100))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(100, 10))

# 손실 함수 선택
loss_fn = nn.CrossEntropyLoss()

# 가중치를 학습하기 위한 최적화 기법 선택
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 총 3번의 에포크 동안 모델 학습
epochs = 3

for epoch in range(epochs):
    for data, targets in loader_train:
        optimizer.zero_grad()      # 옵티마이저의 기울기 초기화
        y_pred = model(data)          # 순전파 연산으로 예측값 계산
        loss = loss_fn(y_pred, targets)  # 손실 함수로 비용 계산
        loss.backward()            # 역전파 연산으로 기울기 계산
        optimizer.step()           # 옵티마이저를 통해 파라미터 업데이트

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch + 1, 3, loss.item()))

model.eval()  # 신경망을 추론 모드로 전환
correct = 0

# 데이터로더에서 미니배치를 하나씩 꺼내 추론을 수행
with torch.no_grad():  # 추론 과정에는 미분이 필요없음
    for data, targets in loader_test:

        outputs = model(data)  # 데이터를 입력하고 출력을 계산

        # 추론 계산
        _, predicted = torch.max(outputs.data, 1)  # 확률이 가장 높은 레이블이 무엇인지 계산
        correct += predicted.eq(targets.data.view_as(predicted)).sum()  # 정답과 일치한 경우 정답 카운트를 증가

# 정확도 출력
data_num = len(loader_test.dataset)  # 데이터 총 건수
print('\n테스트 데이터에서 예측 정확도: {}/{} ({:.0f}%)\n'.format(correct, data_num, 100. * correct / data_num))

index = 2018

model.eval()  # 신경망을 추론 모드로 전환
data = X_test[index]
output = model(data)  # 데이터를 입력하고 출력을 계산
_, predicted = torch.max(output.data, 0)  # 확률이 가장 높은 레이블이 무엇인지 계산

print("예측 결과 : {}".format(predicted))

X_test_show = (X_test[index]).numpy()
plt.imshow(X_test_show.reshape(28, 28), cmap='gray')
print("이 이미지 데이터의 정답 레이블은 {:.0f}입니다".format(y_test[index]))