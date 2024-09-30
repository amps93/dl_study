"""
코사인 유사도(Cosine Similarity)
두 벡터의 방향이 완전히 동일한 경우에는 1의 값을 가지며, 90°의 각을 이루면 0, 180°로 반대의 방향을 가지면 -1의 값을 가짐
코사인 유사도는 -1 이상 1 이하의 값을 가지며 값이 1에 가까울수록 유사도가 높다고 판단
"""

import numpy as np
from numpy import dot
from numpy.linalg import norm


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


doc1 = np.array([0, 1, 1, 1])
doc2 = np.array([1, 0, 1, 1])
doc3 = np.array([2, 0, 2, 2])

print("문서 1과 문서2의 유사도 :", cos_sim(doc1, doc2))
print("문서 1과 문서3의 유사도 :", cos_sim(doc1, doc3))
print("문서 2와 문서3의 유사도 :", cos_sim(doc2, doc3))
