"""
파이토치는 미분을 통해 얻은 기울기를 이전에 계산된 기울기 값에 누적시키는 특징이 있음

기울기를 누적 시키는 이유는 미니배치 때문이다.
1. 미니배치학습은 데이터셋을 작은 배치 단위로 나누어 학습한다.
2. 이 때 각 미니 배치에서 손실함수의 기울기를 계산한 후 누적하여 전체 데이터의 기울기를 구한다
3. 미니 배치 이후 값을 초기화 한다. -> 초기화 하지 않으면 이전 배치의 기울기 값이 누적되어 다음 배치에서 잘못된 기울기가  적용될 수 있기 때문

1000개의 데이터를 100개로 나누어 학습
1. 첫번째 100개 데이터 학습 후 기울기 계산 - 이 때 기울기는 누적된 후 합쳐짐
2. 기울기 초기화
3. 다음 100개 데이터 학습 후 기울기 계산 - 이 때 기울기는 누적된 후 합쳐짐
4. 기울기 초기화
5. 반복

"""
import torch

w = torch.tensor(2.0, requires_grad=True)

nb_epochs = 20

for epoch in range(nb_epochs + 1):
    z = 2 * w

    z.backward()
    print('수식을 w로 미분한 값 :', w.grad)  # 계속해서 미분값인 2가 누적
