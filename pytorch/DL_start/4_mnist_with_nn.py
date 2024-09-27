import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

# 레이블 데이터 타입을 정수형으로 변환
mnist.target = mnist.target.astype(np.int8)

X = mnist.data / 255  # 0-255값을 [0,1] 구간으로 정규화
y = mnist.target

plt.imshow(X[0].reshape(28, 28), cmap='gray')
plt.show()
print("이 이미지 데이터의 레이블은 {:.0f}이다".format(y[0]))
