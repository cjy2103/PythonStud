carList = ["투싼", "토레스", "스포티지"]

print(carList)

print(carList[0], carList[-1])

carList[1] = "셀토스"

print(carList)

print(carList[0:1])
print(carList[1:])

intList = [1, 2, 3, 4, 5, 6, 7, 8]

del intList[0]

print(intList)

del intList[2:6]

print(intList)

mergeList = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

print(mergeList[0][1])

repeat = intList * 3
print(repeat)

print(len(intList))

intList.reverse()

print(intList)

intList.sort()

print(intList)

print("2의 개수?", intList.count(2))

intList.append(9)

print(intList)

intList.extend([10, 11])

print(intList)

intList.insert(2, 4)

print(intList)

intList.remove(4)

print(intList)

intList.pop(2)
intList.pop()

print(intList)

intList.clear()

print(intList)

proList = ["Java", "Kotlin", "JavaScript", "Python", "C"]

print(proList.index("Java"))
print(proList.index("Python", 0, 5))

