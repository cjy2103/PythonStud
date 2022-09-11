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

# zero 행렬 생성
print(np.zeros((3, 3)))

# 유닛 행렬
print(np.ones((2, 4)))

# 모든 요소가 7인 2 * 4 행렬
print(np.full((2, 4), 7))

# 단위 행렬
print(np.eye(3))

# 배열의 차원 변환
a = np.arange(12)
print(a)

b = a.reshape((4, 3))
print(b)

lst = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
]

arr = np.array(lst)

a = arr[0:3, 0:3]
print(a)

a = arr[1:, 1:]
print(a)

# 배열간 연산 기능 #

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 각 요소 더하기
c = a + b

print(c)

# 각 요소 곱하기
c = a * b
print(c)

c = np.multiply(a, b)
print(c)

# 각 요소 나누기
c = a / b
print(c)

c = np.divide(a, b)
print(c)

# 행렬의 곱
array1 = [[1, 2], [3, 4]]
array2 = [[5, 6], [7, 8]]


a = np.array(array1)
b = np.array(array2)

print(a)
print(b)

c = np.dot(a, b)

print(c)

