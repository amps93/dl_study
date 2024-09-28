https://wikidocs.net/61271
# 기울기 소실 (Gradient Vanishing)
역전파 과정에서 입력층으로 갈 수록 기울기(Gradient)가 점차적으로 작아져 가중치들이 업데이트가 제대로 되지 않아 최적의 모델을 찾을 수 없게 됨

# 기울기 폭주 (Gradient Exploding)
기울기가 점차 커지더니 가중치들이 비정상적으로 큰 값이 되면서 결국 발산하는 현상

# 해결 방법
1. ReLU와 ReLU의 변형들
2. 가중치 초기화(Weight initialization)
   1. 세이비어 초기화(Xavier Initialization)
   2. He 초기화(He initialization)
3. 배치 정규화(Batch Normalization)
   1. 내부 공변량 변화(Internal Covariate Shift)
   2. 배치 정규화(Batch Normalization)
   3. 배치 정규화의 한계
      1. 미니 배치 크기에 의존적이다.
      2. RNN에 적용하기 어렵다.
4. 층 정규화(Layer Normalization)