import numpy as np

num = 0
while num < 5:
    np.random.seed(5)
    print(np.random.random())
    num += 1

np.random.seed(5)
num1 = np.random.random()
num2 = np.random.random()
print("num1 = %f, num2 = %f" % (num1, num2))

np.random.seed(5)
num1 = np.random.random()
num2 = np.random.random()
print("num1 = %f, num2 = %f" % (num1, num2))

np.random.seed(1)
X_assess = np.random.randn(2, 3)
print("X_asses:")
print(X_assess)

np.random.seed(1)
X_assess = np.random.randn(2, 3)
print("X_asses:")
print(X_assess)