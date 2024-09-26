"""
로지스틱 회귀가 binary classification 문제 였다면
소프트맥스 회귀는 multiclass classification 문제임
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
"""
1. 파이토치로 소프트맥스의 비용 함수 구현하기 (로우-레벨)
"""
# 소프트 맥스 함수 구현
z = torch.FloatTensor([1, 2, 3])
hypothesis = F.softmax(z, dim=0)
# print(hypothesis)  # tensor([0.0900, 0.2447, 0.6652])
# print(hypothesis.sum())  # tensor(1.)

# (3, 5) 행렬(텐서) 생성
z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
# print(hypothesis)  # 5개의 클래스 중 어떤 클래스가 정답인지를 예측한 결과. 각 1, 1, 3 예측
# tensor([[0.2645, 0.1639, 0.1855, 0.2585, 0.1277],
#         [0.2430, 0.1624, 0.2322, 0.1930, 0.1694],
#         [0.2226, 0.1986, 0.2326, 0.1594, 0.1868]], grad_fn=<SoftmaxBackward>)

# 임의의 레이블 생성
y = torch.randint(5, (3,)).long()
# print(y)  # tensor([0, 2, 1])

# 원핫 인코딩 수행 - 모든 원소가 0의 값을 가진 3 × 5 텐서 생성
y_one_hot = torch.zeros_like(hypothesis)
# print(y_one_hot.scatter_(1, y.unsqueeze(1), 1))  # y.unsqueeze(1)  # (3, ) 크기를 가진 y텐서를 (3, 1) 차원의 텐서로 변형
# tensor([[1., 0., 0., 0., 0.],
#         [0., 0., 1., 0., 0.],
#         [0., 1., 0., 0., 0.]])

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()  # 소프트맥스 공식
print(cost)

# 데이터로 소프트맥스 회귀 구현
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# 원핫 인코딩
y_one_hot = torch.zeros(8, 3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)

# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros((1, 3), requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # 가설
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)

    # 비용 함수
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

"""
2. 파이토치로 소프트맥스의 비용 함수 구현하기 (하이-레벨)
로우-코드에서 개발한 소프트 맥스 코드
z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

1. F.softmax() + torch.log() = F.log_softmax()
앞서 소프트맥스 함수의 결과에 로그를 씌울 때는 다음과 같이 소프트맥스 함수의 출력값을 로그 함수의 입력으로 사용- torch.log(F.softmax(z, dim=1))
파이토치에서는 두 개의 함수를 결합한 F.log_softmax()라는 도구를 제공 - F.log_softmax(z, dim=1)

2. F.log_softmax() + F.nll_loss() = F.cross_entropy()
로우-레벨로 구현한 비용 함수는 다음과 같았음
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()
위의 수식에서 torch.log(F.softmax(z, dim=1))를 방금 배운 F.log_softmax()로 대체할 수 있음
(y_one_hot * - F.log_softmax(z, dim=1)).sum(dim=1).mean()

3. 더 간단하게
F.nll_loss(F.log_softmax(z, dim=1), y)

4. 더 간단하게
F.cross_entropy(z, y)
결국 로우-코드에서 개발한 소프트 맥스 코드와 같은 의미
"""
# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros((1, 3), requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산
    z = x_train.matmul(W) + b
    cost = F.cross_entropy(z, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))


"""
3. 소프트맥스 회귀 nn.Module로 구현하기
"""
# 모델을 선언 및 초기화. 4개의 특성을 가지고 3개의 클래스로 분류. input_dim=4, output_dim=3.
model = nn.Linear(4, 3)

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

"""
4. 소프트맥스 회귀 클래스로 구현하기
"""
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3) # Output이 3!

    def forward(self, x):
        return self.linear(x)


model = SoftmaxClassifierModel()

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))