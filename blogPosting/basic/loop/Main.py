lst = [1, 2, 3, 4, 5]

for i in lst:
    print(i)

print("############################")

for i in range(10):
    print(i)


print("############################")

i = 1
while i <= 10:
    print(i, "출력 되었습니다.")
    i += 1

print("############################")

i = 1
while i <= 20:
    if i == 5:
        break
    print(i, "출력 되었습니다.")
    i += 1

print("############################")

i = 1

while i <= 10:
    i += 1
    if i % 2 == 1:
        continue
    print(i, "출력 되었습니다.")

