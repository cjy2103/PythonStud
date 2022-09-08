a = (1, 2, 3, "사과", "바나나")
b = (1,)
c = ()
print(a)
print(b)
print(c)

print(a[2])
print(a[3])

print(a[2:5])
print(a[4:])

print("길이", len(a))

d = a + b

print(d)

d = a * 3

print(d)

if 2 in a:
    print("1 in a")
else:
    print("1 not in a")

if 10 in a:
    print("10 in a")
else:
    print("10 not in a")

print(a.count(2))

a = 1, 2, 3, 4, 5

print(a)

a = ((1, 2, 3), (4, 5, 6), (7, 8, 9))

print(a)

a = (1, 2, 3, 4, 5)

a1 = list(a)

print(a1)

a2 = tuple(a1)

print(a2)
