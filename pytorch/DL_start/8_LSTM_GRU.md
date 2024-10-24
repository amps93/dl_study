# RNN의 한계
1. 장기 의존성 문제
2. 기울기 소실/폭발 문제

이를 극복하기 위해 LSTM 등장

# LSTM(Long Short-Term Memory)
RNN(Recurrent Neural Network)의 한 종류로, 장기 의존성 문제(long-term dependency)를 해결하기 위해 고안된 모델

## LSTM의 세 가지 게이트와 셀 상태(cell state)
1. 셀 상태(Cell State)\
셀 상태는 시퀀스의 각 타임스텝에서 정보가 흘러가는 경로로, 정보를 그대로 전달하거나 수정
LSTM은 중요한 정보를 셀 상태에 보존하고, 필요 없는 정보를 제거하는 역할을 함

2. 게이트(Gates)\
게이트는 정보가 셀 상태로 들어가거나 나가는 것을 제어하는 매개체로,
각각의 게이트는 시그모이드(sigmoid) 함수를 통해 0과 1 사이의 값을 출력하며, 정보가 얼마나 흘러갈지 결정한다
   * 입력 게이트(Input Gate): 새로운 정보를 셀 상태에 얼마나 업데이트할지를 결정합니다.
   * 망각 게이트(Forget Gate): 셀 상태에서 불필요한 정보를 얼마나 잊을지를 결정합니다.
   * 출력 게이트(Output Gate): 셀 상태로부터 출력할 정보(은닉 상태)를 결정합니다.

## LSTM 동작 과정
1. 망각 게이트: 이전 셀 상태에서 어떤 정보를 잊을지 결정
2. 입력 게이트: 새로운 정보를 셀 상태에 얼마나 추가할지 결정
3. 셀 상태 업데이트: 망각 게이트에서 잊기로 한 부분을 제외한 나머지와, 새로운 정보를 기반으로 셀 상태를 업데이트
4. 출력 게이트: 은닉 상태를 업데이트하여 다음 타임스텝으로 전달할 값을 결정

## 파이토치에서 사용 방법
```nn.LSTM(input_dim, hidden_size ,batch_first=True)```

# GRU(Gated Recurrent Unit)
LSTM에서는 출력, 입력, 삭제 게이트라는 3개의 게이트가 존재한 반면, GRU에서는 업데이트 게이트와 리셋 게이트 두 가지 게이트만이 존재함
기존에 LSTM을 사용하면서 최적의 하이퍼파라미터를 찾아낸 상황이라면 굳이 GRU로 바꿔서 사용할 필요는 없음

## 파이토치에서 사용 방법
```nn.GRU(input_dim, hidden_size, batch_first=True)  ```

