a = 10

if a >= 10:
    print("10보다 큼")
else:
    print("10보다 작음")

b = "python"
if b == "python":
    print("맞음")
else:
    print("아님")

c = [1, 2, 3, 4, 5]

if 1 in c:
    print("리스트에 1 포함")
elif 6 in c:
    print("리스트에 6 포함")
elif 10 not in c:
    print("리스트에 10 미포함")
