import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 모델 초기화
W = torch.zeros(1, requires_grad=True)  # requires_grad 해당 텐서에서 연산이 수행될 때, 그 텐서에 대한 기울기를 계산할 수 있도록 설정
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
opt = optim.SGD([W, b], lr=0.01)  # lr: 학습률

nb_epochs = 1999
for epoch in range(nb_epochs + 1):
    h0 = x_train * W + b
    # cost 계산
    cost = torch.mean((h0 - y_train) ** 2)

    # cost로 H(x) 개선
    opt.zero_grad()  # gradient 초기화
    cost.backward()  # 비용 함수 미분하여 gradient 계산
    opt.step()  # W, b 업데이트

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))



