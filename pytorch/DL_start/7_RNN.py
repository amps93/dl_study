'''
https://wikidocs.net/60690
Feed Forward Neural Network: 은닉층에서 활성화 함수를 지난 값은 오직 출력층 방향으로만 진행
Recurrent Neural Network: 은닉층의 노드에서 활성화 함수를 통해 나온 결과값을 출력층 방향으로도 보내면서, 다시 은닉층 노드의 다음 계산의 입력으로 보내는 특징을 갖고있음

RNN 구조
* 입력: 입력 시퀀스는 x1, x2, x2 ... 처럼 시간 순서에 따라 입력됨
* 은닉 상태: 각 타임스텝 t에서 은닉상태 h_t는 이전의 은닉 상태 h_(t-1)과 현재 입력 x_t를 기반으로 계산됨
    * ht = f(W_hh*h_(t-1) + W_xh*W_xt)
    * 여기서 f는 활성화 함수 (주로 tanh, ReLU), W_hh와 W_xt는 각각 은닉 상태와 입력에 대한 가중치 행렬
* 출력: 은닉 상태를 기반으로 최종 출력 y_t를 계산
    * y_t=g(W_hx*h_t)
    * 여기서 W_hy는 출력에 대한 가중치 행렬, g는 출력 함수

* pseudocode로 구현
# 아래의 코드는 의사 코드(pseudocode)로 실제 동작하는 코드가 아님.
hidden_state_t = 0 # 초기 은닉 상태를 0(벡터)로 초기화
for input_t in input_length: # 각 시점마다 입력을 받는다.
    output_t = tanh(input_t, hidden_state_t) # 각 시점에 대해서 입력과 은닉 상태를 가지고 연산
    hidden_state_t = output_t # 계산 결과는 현재 시점의 은닉 상태가 된다.
'''

"""
파이토치의 nn.RNN()
"""
import torch
import torch.nn as nn

input_size = 5  # 입력의 크기
hidden_size = 8  # 은닉 상태의 크기

# (batch_size, time_steps, input_size)
# 배치 크기는 1, 10번의 시점동안 5차원의 입력 벡터가 들어가도록 텐서를 정의
inputs = torch.Tensor(1, 10, 5)

# cell 생성
cell = nn.RNN(input_size, hidden_size, batch_first=True)

outputs, _status = cell(inputs)

print(outputs.shape)  # 모든 time-step의 hidden_state
print(_status.shape)  # 최종 time-step의 hidden_state

"""
깊은 순환 신경망(Deep Recurrent Neural Network)
"""
# (batch_size, time_steps, input_size)
inputs = torch.Tensor(1, 10, 5)

#  nn.RNN()의 인자인 num_layers에 값을 전달하여 층을 쌓음
cell = nn.RNN(input_size=5, hidden_size=8, num_layers=2, batch_first=True)

print(outputs.shape)  # 모든 time-step의 hidden_state
print(_status.shape)  # (층의 개수, 배치 크기, 은닉 상태의 크기)
# 두번째 리턴값의 크기는 층이 1개였던 RNN 셀 때와 달라졌는데, 여기서 크기는 (층의 개수, 배치 크기, 은닉 상태의 크기)에 해당됨

"""
양방향 순환 신경망(Bidirectional Recurrent Neural Network)
시점 t에서의 출력값을 예측할 때 이전 시점의 데이터뿐만 아니라, 이후 데이터로도 예측할 수 있다는 아이디어에 기반

양방향 RNN은 하나의 출력값을 예측하기 위해 기본적으로 두 개의 메모리 셀을 사용
1. 첫번째 메모리 셀은 앞에서 배운 것처럼 앞 시점의 은닉 상태(Forward States)를 전달받아 현재의 은닉 상태를 계산
2. 두번째 메모리 셀은 앞 시점의 은닉 상태가 아니라 뒤 시점의 은닉 상태(Backward States)를 전달 받아 현재의 은닉 상태를 계산
"""
# (batch_size, time_steps, input_size)
inputs = torch.Tensor(1, 10, 5)

cell = nn.RNN(input_size=5, hidden_size=8, num_layers=2, batch_first=True, bidirectional=True)
outputs, _status = cell(inputs)
print(outputs.shape) # (배치 크기, 시퀀스 길이, 은닉 상태의 크기 x 2)
print(_status.shape) # (층의 개수 x 2, 배치 크기, 은닉 상태의 크기)

