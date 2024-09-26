"""
# 1. 미니 배치와 배치 크기

1. 미니 배치 학습을 하게되면 미니 배치만큼만 가져가서 미니 배치에 대한 대한 비용(cost)를 계산하고, 경사 하강법을 수행
2. 다음 미니 배치를 가져가서 경사 하강법을 수행하고 마지막 미니 배치까지 이를 반복
3. 전체 데이터에 대한 학습이 1회 끝나면 1 에포크(Epoch)가 끝나게 됨

전체 데이터에 대해서 한 번에 경사 하강법을 수행하는 방법: '배치 경사 하강법'
미니 배치 단위로 경사 하강법을 수행하는 방법: '미니 배치 경사 하강법'

배치 경사 하강법은 경사 하강법을 할 때, 전체 데이터를 사용하므로 가중치 값이 최적값에 수렴하는 과정이 매우 안정적이지만, 계산량이 너무 많이 듦
미니 배치 경사 하강법은 경사 하강법을 할 때, 전체 데이터의 일부만을 보고 수행하므로 최적값으로 수렴하는 과정에서 값이 조금 헤매기도 하지만 훈련 속도가 빠름
배치 크기는 보통 2의 제곱수를 사용 ex) 2, 4, 8, 16, 32, 64


# 2.이터레이션(Iteration)
한 번의 에포크 내에서 이루어지는 매개변수인 가중치 W와 b의 업데이트 횟수
전체 데이터가 2,000일 때 배치 크기를 200으로 한다면 이터레이션의 수는 총 10개
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset  # 텐서데이터셋
from torch.utils.data import DataLoader  # 데이터로더

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

dataset = TensorDataset(x_train, y_train)
# 데이터로더는 기본적으로 (데이터셋, 미니배치크기) 2개의 인자를 입력받음
# shuffle=True - epoch마다 데이터셋을 섞어 학습
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# model, optimizer 정의
model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        print('batch_idx =', batch_idx)
        print('samples =', samples)
        x_train, y_train = samples
        prediction = model(x_train)

        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx + 1, len(dataloader), cost.item())
        )

# 임의의 입력 [73, 80, 75]를 선언
new_var = torch.FloatTensor([[73, 80, 75]])
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var)
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)
