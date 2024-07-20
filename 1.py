import numpy as np

# Данные
X = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
y = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

# Средние значения
X_mean = np.mean(X)
y_mean = np.mean(y)

# Ковариация и дисперсия
cov_xy = np.mean((X - X_mean) * (y - y_mean))
var_X = np.mean((X - X_mean) ** 2)

# Коэффициенты с интерсептом
beta1 = cov_xy / var_X
beta0 = y_mean - beta1 * X_mean

# Коэффициент без интерсепта
beta1_no_intercept = np.sum(X * y) / np.sum(X ** 2)

# Вывод моделей
print(f"Модель линейной регрессии с интерсептом: y = {beta0:.2f} + {beta1:.2f}X")
print(f"Модель линейной регрессии без интерсепта: y = {beta1_no_intercept:.2f}X")
