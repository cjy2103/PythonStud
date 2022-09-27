class Calculator:
    # 생성자 _init_은 반드시 선언해 주어야 함
    def __init__(self):
        self.result = 0

    def add(self, num):
        self.result += num
        return self.result

    def mul(self, num):
        self.result *= num
        return self.result

    def minus(self, num):
        self.result -= num
        return self.result

    def div(self, num):
        self.result /= num
        return self.result


cal = Calculator()

print(cal.add(5))
print(cal.mul(3))
print(cal.minus(3))
print(cal.div(4))
