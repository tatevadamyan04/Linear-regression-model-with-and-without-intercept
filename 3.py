import numpy as np

# Данные
X = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
y = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

# Гиперпараметры
learning_rate = 0.0001  # Попробуйте уменьшить коэффициент обучения
num_iterations = 10000  # Количество итераций
tolerance = 1e-6  # Порог сходимости

# Инициализация коэффициентов
beta0 = 0
beta1 = 0
prev_loss = float('inf')

# Градиентный спуск
n = len(X)
for i in range(num_iterations):
    # Вычисление градиентов
    gradient_beta0 = -2 / n * np.sum(y - (beta0 + beta1 * X))
    gradient_beta1 = -2 / n * np.sum(X * (y - (beta0 + beta1 * X)))
    
    # Обновление коэффициентов
    beta0 -= learning_rate * gradient_beta0
    beta1 -= learning_rate * gradient_beta1
    
    # Вычисление потерь
    loss = np.mean((y - (beta0 + beta1 * X)) ** 2)
    
    # Проверка на сходимость
    if abs(prev_loss - loss) < tolerance:
        print(f"Сходимость достигнута после {i+1} итераций.")
        break
    
    prev_loss = loss
    
    # Отладочные сообщения
    if i % 1000 == 0:  # Каждые 1000 итераций
        print(f"Итерация {i+1}: beta0 = {beta0:.2f}, beta1 = {beta1:.2f}, loss = {loss:.2f}")

# Вывод результата
print(f"Коэффициент линейной регрессии с интерсептом: y = {beta0:.2f} + {beta1:.2f}X")
