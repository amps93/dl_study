과적합(Overfitting) 막는 법

1. 데이터의 양을 늘리기
2. 모델의 복잡도 줄이기
   1. 아래 인공 신경망은 3개의 레이어를 가지고 있음
   2. 아래 인공 신경망은 2개의 레이어를 가지고 있음. 위의 신경망이 과적합 현상을 보이면 아래와 같이 인공 신경망의 복잡도를 줄일 수 있음
    ```
    # 3개의 레이어를 가지고 있는 인공 신경망
    class Architecture1(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(Architecture1, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.relu = nnReLU()
            self.fc3 = nn.Linear(hidden_size, num_classes)
            
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.relu(out)
            out = self.fc3(out)
            return out
    ```
    ```
    # 2개의 레이어를 가지고 있는 인공 신경망
    class Architecture1(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(Architecture1, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)
    
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out
    ```
3. 가중치 규제 적용하기
   1. 복잡한 모델을 좀 더 간단하게 하는 방법으로 가중치 규제(Regularizaiton)가 있음
       * L1 규제 : 가중치 w들의 절대값 합계를 비용 함수에 추가. L1 노름이라고도 함.
       * L2 규제 : 모든 가중치 w들의 제곱합을 비용 함수에 추가. L2 노름이라고도 함.
    ```
    model = Architecture1(10, 20, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    ```
4. 드롭아웃
   1. 학습 과정에서 신경망의 일부를 사용하지 않는 방법
   2. 드롭아웃의 비율을 0.5로 한다면 학습 과정마다 랜덤으로 절반의 뉴런을 사용하지 않고, 절반의 뉴런만을 사용
   3. 드롭아웃은 신경망 학습 시에만 사용하고, 예측 시에는 사용하지 않는 것이 일반적