import math



x = [1, 2]
y = [2, 5]

xs = 0
ys = 0

SumOfDeviation  = 0.0
SumOfVariance = 0.0
SumOfStandardDeviation = 0.0



for i, x in enumerate(xs):
    deviation = ys[i] - y[i]  # 편차
    variance = deviation ** 2 # 분산
    standardDeviation = math.sqrt(variance)   # 표준편차

    SumOfDeviation += deviation
    SumOfVariance += variance
    SumOfStandardDeviation += standardDeviation

    print("X가 ", x, "일때, 실제값:" , ys[i], ", 예상값:" , y[i])
    print(" 편차(잔차) :", deviation )
    print(" 분산 :", variance)
    print(" 표준편차(잔차) :", standardDeviation)

print('편차의 합:' , SumOfDeviation)
print('분산의 합:' , SumOfVariance)
print('표준편차의 합:' , SumOfStandardDeviation)