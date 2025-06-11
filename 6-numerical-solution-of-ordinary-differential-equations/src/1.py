from math import exp, sin, cos

import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate

# Максимально допустимое число точек и итераций для адаптивного уточнения
MAX_N = 100_000
MAX_ITERATIONS = 1_000_000


# Класс для хранения результата решения ОДУ
class Result:
    def __init__(self, x, y, error):
        self.x = x  # Список узлов сетки
        self.y = y  # Список значений решения в узлах
        self.error = error  # Оценка погрешности
        self.n = len(x)  # Число точек


# Решатель для одношаговых методов с оценкой ошибки по правилу Рунге
def solve_one_step(equation, method, p, x0, xn, n, y0, eps):
    error = float('inf')
    # Инициализируем сетку по x равномерным разбиением
    x = [x0 + i * (xn - x0) / (n - 1) for i in range(n)]
    # Один прогон метода
    try:
        y = method(equation, x, y0)
    except OverflowError:
        raise Exception('Не удалось вычислить, слишком большие значения')

    # Адаптивное удвоение числа точек до достижения точности eps
    while error > eps:
        if n > MAX_N:
            raise Exception(f'Слишком много точек ({n}), точность не достигнута')
        n *= 2
        x = [x0 + i * (xn - x0) / (n - 1) for i in range(n)]
        try:
            next_y = method(equation, x, y0)
        except OverflowError:
            raise Exception('Не удалось вычислить, слишком большие значения')
        # Оценка локальной погрешности по правилу Рунге (разность на последнем узле)
        error = abs(next_y[-1] - y[-1]) / (2 ** p - 1)
        y = next_y

    return Result(x, y, error)


# Решатель для многошаговых методов: сравниваем с точным решением
def solve_multy_step(equation, method, solution, x0, xn, n, y0, eps):
    error = float('inf')
    x = [x0 + i * (xn - x0) / (n - 1) for i in range(n)]
    try:
        y = method(equation, x, y0, eps)
    except OverflowError:
        raise Exception('Не удалось вычислить, слишком большие значения')

    # Адаптивное удвоение числа точек до достижения точности eps
    while error > eps:
        if n > MAX_N:
            raise Exception(f'Слишком много точек ({n}), точность не достигнута')
        n *= 2
        x = [x0 + i * (xn - x0) / (n - 1) for i in range(n)]
        try:
            y = method(equation, x, y0, eps)
        except OverflowError:
            raise Exception('Не удалось вычислить, слишком большие значения')
        # Оценка глобальной погрешности: макс. отклонение от точного решения
        error = max(abs(solution(xi, x0, y0) - yi) for xi, yi in zip(x, y))

    return Result(x, y, error)


# Генерация текстового лога с табличным выводом
def generate_result_log(method_name, result, y_real):
    log = [method_name]
    table = [
        ['x'] + result.x,
        ['y числ.'] + result.y,
        ['y точн.'] + y_real
    ]
    log.append(tabulate(table, tablefmt='grid'))
    log.append(f'Погрешность: {result.error}')
    log.append(f'Число точек: {result.n}')
    return "\n".join(log)


# Печать серии логов с разделителями
def print_result(result_log):
    sep = '=' * 50
    print(f"\n{sep}\n")
    print(f"\n{sep}\n".join(result_log))
    print(f"\n{sep}\n")


# Меню выбора из списка опций
def choose_options(message, options):
    print(f"{message}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    choice = None
    while choice is None:
        try:
            val = int(input())
            if 1 <= val <= len(options):
                choice = val - 1
            else:
                print("Выберите корректный номер")
        except ValueError:
            print("Введите число")
    return choice


# Чтение положительного целого
def read_positive_integer(message):
    val = None
    while val is None:
        try:
            x = int(input(f"{message}: "))
            if x > 0:
                val = x
            else:
                print("Должно быть > 0")
        except ValueError:
            print("Введите целое число")
    return val


# Чтение вещественного числа
def read_float(message):
    val = None
    while val is None:
        try:
            val = float(input(f"{message}: "))
        except ValueError:
            print("Введите число")
    return val


# Проверка, что точное решение определено на всём интервале
def is_safe_interval(x0, xn, f):
    for x in np.linspace(x0, xn, int((xn - x0) * 100) + 1):
        try:
            f(x)
        except Exception:
            return False
    return True


# Отрисовка графика численного и точного решения
def draw_plot(x, y, solution_f, method_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label='Численное', s=20)
    xs = np.linspace(min(x), max(x), 500)
    ys = np.array([solution_f(xi) for xi in xs])
    plt.plot(xs, ys, label='Точное')
    plt.title(method_name)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


# --- Определение численных методов ---

def euler_method(f, x, y0):
    """Метод Эйлера: y_{k+1} = y_k + h * f(x_k,y_k)"""
    y = [y0]
    for i in range(1, len(x)):
        h = x[i] - x[i - 1]
        y.append(y[-1] + h * f(x[i - 1], y[-1]))
    return y


def improved_euler_method(f, x, y0):
    """Усоверш. Эйлера (Heun): усреднение f в концах шага"""
    y = [y0]
    for i in range(1, len(x)):
        h = x[i] - x[i - 1]
        k1 = f(x[i - 1], y[-1])
        k2 = f(x[i - 1] + h, y[-1] + h * k1)
        y.append(y[-1] + (h / 2) * (k1 + k2))
    return y


def second_order_runge_kutta_method(f, x, y0):
    """RK2: 2 оценки k1, k2"""
    y = [y0]
    for i in range(1, len(x)):
        h = x[i] - x[i - 1]
        k1 = f(x[i - 1], y[-1])
        k2 = f(x[i - 1] + h, y[-1] + h * k1)
        y.append(y[-1] + h * (k1 + k2) / 2)
    return y


def fourth_order_runge_kutta_method(f, x, y0):
    """RK4: k1..k4, взвешенное среднее"""
    y = [y0]
    for i in range(1, len(x)):
        h = x[i] - x[i - 1]
        k1 = f(x[i - 1], y[-1])
        k2 = f(x[i - 1] + h / 2, y[-1] + h * k1 / 2)
        k3 = f(x[i - 1] + h / 2, y[-1] + h * k2 / 2)
        k4 = f(x[i - 1] + h, y[-1] + h * k3)
        y.append(y[-1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
    return y


def adams_method(f, x, y0, eps):
    """Метод Адамса: предиктор (явный), корректор (неявный)"""
    n = len(x)
    # первые 4 точки через RK4
    y = fourth_order_runge_kutta_method(f, x[:4], y0)
    for i in range(4, n):
        h = x[i] - x[i - 1]
        # predictor: Адамс-Бэшфорд 4-го порядка
        y_pred = y[-1] + h / 24 * (
                55 * f(x[i - 1], y[-1]) - 59 * f(x[i - 2], y[-2]) + 37 * f(x[i - 3], y[-3]) - 9 * f(x[i - 4],
                                                                                                    y[-4]))
        # corrector: итерации до погрешности eps
        for it in range(MAX_ITERATIONS):
            f_pred = f(x[i], y_pred)
            y_corr = y[-1] + h / 24 * (
                    9 * f_pred + 19 * f(x[i - 1], y[-1]) - 5 * f(x[i - 2], y[-2]) + f(x[i - 3], y[-3]))
            if abs(y_corr - y_pred) < eps:
                y_pred = y_corr
                break
            y_pred = y_corr
        else:
            raise Exception("Метод Адамса: не сошелся корректор")
        y.append(y_pred)
    return y


def milne_method(f, x, y0, eps):
    """Метод Милна: предиктор-корректор с 4 историческими точками"""
    n = len(x)
    y = fourth_order_runge_kutta_method(f, x[:4], y0)
    for i in range(4, n):
        h = x[i] - x[i - 1]
        # predictor: Милн (явная формула)
        y_pred = y[-4] + 4 * h / 3 * (2 * f(x[i - 3], y[-3]) - f(x[i - 2], y[-2]) + 2 * f(x[i - 1], y[-1]))
        # corrector loop
        for it in range(MAX_ITERATIONS):
            f_pred = f(x[i], y_pred)
            y_corr = y[-2] + h / 3 * (f(x[i - 2], y[-2]) + 4 * f(x[i - 1], y[-1]) + f_pred)
            if abs(y_corr - y_pred) < eps:
                y_pred = y_corr
                break
            y_pred = y_corr
        else:
            raise Exception("Метод Милна: не сошелся корректор")
        y.append(y_pred)
    return y


# Список доступных уравнений и их точных решений
EQUATIONS = [
    lambda x, y: y + (1 + x) * y ** 2,
    lambda x, y: x + y,
    lambda x, y: cos(x) - y
]

EQUATIONS_SOLUTIONS = [
    lambda x, x0, y0: -exp(x) / (x * exp(x) - (x0 * exp(x0) * y0 + exp(x0)) / y0),
    lambda x, x0, y0: exp(x - x0) * (y0 + x0 + 1) - x - 1,
    lambda x, x0, y0: (y0 - sin(x0)) * exp(-(x - x0)) + sin(x)
]


def main():
    # Выбор уравнения
    idx = choose_options('Выберите уравнение', EQUATIONS_NAMES)
    f = EQUATIONS[idx]
    sol = EQUATIONS_SOLUTIONS[idx]

    # Чтение параметров задачи
    x0 = read_float('x0')
    xn = read_float('xn')
    if x0 > xn:
        x0, xn = xn, x0
        print('Поменяли x0 и xn')
    n = read_positive_integer('Число точек n')
    y0 = read_float('y0')

    # Проверка области определения решения
    if not is_safe_interval(x0, xn, lambda xx: sol(xx, x0, y0)):
        raise Exception('Решение не определено на всём интервале')

    eps = read_float('Точность eps')
    logs = []

    # Прогоны одношаговых методов
    p_list = [1, 2, 2, 4]
    for meth, name, p in zip(
            [euler_method, improved_euler_method, second_order_runge_kutta_method, fourth_order_runge_kutta_method],
            ['Эйлер', 'Улучш. Эйлер', 'RK2', 'RK4'],
            p_list):
        try:
            res = solve_one_step(f, meth, p, x0, xn, n, y0, eps)
            y_real = [sol(xi, x0, y0) for xi in res.x]
            logs.append(generate_result_log(name, res, y_real))
            draw_plot(res.x, res.y, lambda xx: sol(xx, x0, y0), name)
        except Exception as e:
            logs.append(f"{name}: ошибка {e}")

    # Прогоны многошаговых методов
    for meth, name in [(adams_method, 'Адамс'), (milne_method, 'Милн')]:
        try:
            res = solve_multy_step(f, meth, sol, x0, xn, n, y0, eps)
            y_real = [sol(xi, x0, y0) for xi in res.x]
            logs.append(generate_result_log(name, res, y_real))
            draw_plot(res.x, res.y, lambda xx: sol(xx, x0, y0), name)
        except Exception as e:
            logs.append(f"{name}: ошибка {e}")

    # Вывод финального отчёта
    print_result(logs)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Ошибка: {e}")
