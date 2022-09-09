import numpy as np

# 리스트에서 행렬 생성
data = [[1, 2, 3], [4, 5, 6]]

print(data)

arr = np.array(data)

print(arr)

# 배열의 열수
print(arr.ndim)

# 배열의 차원
print(arr.shape)

# 배열의 원소 접근
print(arr[0][0])

# 1씩 증가하는 배열 생성
print(np.arange(5))

# 1씩 증가하는 배열 생성 ( 시작위치 2)
print(np.arange(2, 10))
