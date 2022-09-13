import matplotlib.pyplot as plt
import numpy as np

a = [1, 2, 3, 4]

plt.plot(a)
# plt.show()

# 다양한 그래프 그리기
t = np.arange(0., 5., 0.2)

plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')
# plt.show()

# 범례 표시
plt.plot([1, 2, 3, 4], [2, 3, 5, 10], 'g', label='Price ($)')
plt.plot([1, 2, 3, 4], [3, 5, 9, 7], 'b', label='Demand (#)')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
# plt.legend(loc='best')          # ncol = 1
plt.legend(loc='best', ncol=2)  # ncol = 2

plt.show()

# 막대 그래프 사용
x = np.arange(3)
years = ['2020', '2021', '2022']
values = [50, 300, 650]
colors = ['y', 'b', 'g']

plt.bar(x, values, color=colors, width=0.6)
plt.xticks(x, years)

plt.show()

# 수평 막대 그래프 사용
y = np.arange(3)
years = ['2018', '2019', '2020']
values = [100, 400, 900]
colors = ['y', 'b', 'g']

plt.barh(y, values, color=colors, height=0.6)
plt.yticks(y, years)

plt.show()

# 파이 차트 그리기

ratio = [34, 32, 16, 18]
labels = ['Apple', 'Banana', 'Melon', 'Grapes']
explode = [0.05, 0.05, 0.05, 0.05]
colors = ['#ff9999', '#ffc000', '#8fd9b6', '#d395d0']
wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}

plt.pie(ratio, labels=labels, autopct='%.1f%%', startangle=260, counterclock=False,
        colors=colors, wedgeprops=wedgeprops)


plt.show()
