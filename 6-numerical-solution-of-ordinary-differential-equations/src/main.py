from math import exp, sin, cos

import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate


def solve_one_step(equation, method, p, x0, xn, n, y0, eps):
    error = 1e9
    x = [x0 + _ * (xn - x0) / (n - 1) for _ in range(n)]
    try:
        y = method(equation, x, y0)
    except OverflowError:
        raise Exception('Не удалось вычислить, слишком большие значения')

    while error > eps:
        if n > MAX_N:
            raise Exception(f'Произведено разбиение на {n} отрезков, но необходимая точность не достигнута')

        n *= 2

        x = [x0 + _ * (xn - x0) / (n - 1) for _ in range(n)]
        try:
            next_y = method(equation, x, y0)
        except OverflowError:
            raise Exception('Не удалось вычислить, слишком большие значения')

        error = abs(next_y[-1] - y[-1]) / (2 ** p - 1)

        y = next_y

    return Result(x, y, error)


def solve_multy_step(equation, method, solution, x0, xn, n, y0, eps):
    error = 1e9
    x = [x0 + _ * (xn - x0) / (n - 1) for _ in range(n)]
    try:
        y = method(equation, x, y0, eps)
    except OverflowError:
        raise Exception('Не удалось вычислить, слишком большие значения')

    while error > eps:
        if n > MAX_N:
            raise Exception(f'Произведено разбиение на {n} отрезков, но необходимая точность не достигнута')

        n *= 2

        x = [x0 + _ * (xn - x0) / (n - 1) for _ in range(n)]
        try:
            y = method(equation, x, y0, eps)
        except OverflowError:
            raise Exception('Не удалось вычислить, слишком большие значения')

        error = max([abs(solution(x_, x0, y0) - y_) for x_, y_ in zip(x, y)])

    return Result(x, y, error)


class Result:
    def __init__(self, x, y, error):
        self.x = x
        self.y = y
        self.error = error
        self.n = len(x)


# I/O
def generate_result_log(method_name, result, y_real):
    log = [method_name]
    table = [
        ['x'] + result.x,
        ['y'] + result.y,
        ['y_real'] + y_real
    ]
    log.append(tabulate(table, tablefmt='grid'))
    log.append(f'Погрешность: {result.error}')
    log.append(f'Для достижения необходимой точности потребовалось разбиение на {result.n} точек')
    return "\n".join(log)


def print_result(result_log):
    separator = '=' * 50
    line = f"\n\n{separator}\n\n".join(result_log)
    print(f'\n{separator}\n\n{line}\n\n{separator}\n')


def choose_options(message, options):
    options_str = ''.join(f'{i + 1} -> {val}\n' for i, val in enumerate(options))[:-1]
    print(f'{message}:\n{options_str}')
    result = None
    while result is None:
        try:
            result = int(input())
            if result not in range(1, len(options) + 1):
                print(f'Выберите один из вариантов:\n{options_str}')
                result = None
                continue
            break
        except:
            print('Значение должно быть числом. Попробуйте снова')
    return result


def read_positive_integer(message):
    value = None
    while value is None:
        try:
            value = int(input(f'{message}: '))
            if value <= 0:
                print('Значение должно быть > 0')
                value = None
        except:
            print('Значение должно быть целым числом!')
            value = None
    return value


def read_float(message):
    value = None
    while value is None:
        try:
            value = float(input(f'{message}: '))
        except:
            print('Значение должно быть целым или дробным числом!')
            value = None
    return value


# validators
def is_save_interval(x0, xn, f):
    for x in np.linspace(x0, xn, int((xn - x0) * 1000)):
        try:
            f(x)
        except Exception:
            return False
    return True


# drawer
def draw_plot(x, y, solution_f, method_name):
    plt.figure(figsize=(8, 6))

    plt.scatter(x, y, color='red', label='Численное решение')

    x_smooth = np.linspace(min(x), max(x), 1000)
    y_smooth = np.array(list(map(solution_f, x_smooth)))
    plt.plot(x_smooth, y_smooth, label='Истинное решение')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(method_name)
    plt.grid(True)
    plt.tight_layout()

    plt.show()


# METHODS

# Максимально допустимое число точек и итераций для адаптивного уточнения
MAX_N = 100_000
MAX_ITERATIONS = 1_000_000


def euler_method(f, x, y0):
    """
    Метод Эйлера (явный, одношаговый).
    Аппроксимация: y_{k+1} = y_k + h * f(x_k, y_k)
    """
    n = len(x)
    y = [y0]  # начальное условие
    for i in range(1, n):
        # шаг вдоль оси x (можно работать и на неравномерной сетке)
        h = x[i] - x[i - 1]
        # обновляем y по формуле Эйлера
        y_next = y[i - 1] + h * f(x[i - 1], y[i - 1])
        y.append(y_next)
    return y


def improved_euler_method(f, x, y0):
    """
    Усовершенствованный метод Эйлера (Heun).
    Аппроксимация: среднее значение скорости на конце и начале шага.
    """
    n = len(x)
    y = [y0]
    for i in range(1, n):
        h = x[i] - x[i - 1]
        # первая оценка производной в начале шага
        k1 = f(x[i - 1], y[i - 1])
        # предсказание y на следующем узле
        y_pred = y[i - 1] + h * k1
        # оценка производной в конце шага
        k2 = f(x[i - 1] + h, y_pred)
        # усреднённое приращение
        y_next = y[i - 1] + (h / 2) * (k1 + k2)
        y.append(y_next)
    return y


def second_order_runge_kutta_method(f, x, y0):
    """
    Метод Рунге–Кутты 2-го порядка (двухточечная схема).
    Аппроксимация: y_{k+1} = y_k + (h/2)*(k1 + k2),
      где k1 = f(x_k,y_k), k2 = f(x_k + h, y_k + h*k1)
    """
    n = len(x)
    y = [y0]
    for i in range(1, n):
        h = x[i] - x[i - 1]
        k1 = f(x[i - 1], y[i - 1])
        k2 = f(x[i - 1] + h, y[i - 1] + h * k1)
        y_next = y[i - 1] + (h / 2) * (k1 + k2)
        y.append(y_next)
    return y


def fourth_order_runge_kutta_method(f, x, y0):
    """
    Метод Рунге–Кутты 4-го порядка.
    Формулы:
      k1 = f(x_k,       y_k)
      k2 = f(x_k + h/2, y_k + h*k1/2)
      k3 = f(x_k + h/2, y_k + h*k2/2)
      k4 = f(x_k + h,   y_k + h*k3)
      y_{k+1} = y_k + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    """
    n = len(x)
    y = [y0]
    for i in range(1, n):
        h = x[i] - x[i - 1]
        xk = x[i - 1]
        yk = y[i - 1]

        k1 = f(xk, yk)
        k2 = f(xk + h / 2, yk + h * k1 / 2)
        k3 = f(xk + h / 2, yk + h * k2 / 2)
        k4 = f(xk + h, yk + h * k3)

        y_next = yk + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y.append(y_next)
    return y


def adams_method(f, x, y0, eps):
    """
    Метод Адамса (предиктор–корректор, порядок 4).
    1) Старт: первые 4 точки считаем RK4.
    2) Predictor (Adams–Bashforth): явная формула с 4 предыдущими f.
    3) Corrector (Adams–Moulton): неявная итерация до |Δ|<eps.
    """
    n = len(x)
    # 1) Разгонные шаги
    y = fourth_order_runge_kutta_method(f, x[:4], y0)

    # 2) Основной цикл по узлам
    for i in range(4, n):
        h = x[i] - x[i - 1]
        # предсказание y на шаге i
        y_pred = (
                y[i - 1]
                + (h / 24) * (
                        55 * f(x[i - 1], y[i - 1])
                        - 59 * f(x[i - 2], y[i - 2])
                        + 37 * f(x[i - 3], y[i - 3])
                        - 9 * f(x[i - 4], y[i - 4])
                )
        )

        # 3) Итерационная коррекция
        for it in range(MAX_ITERATIONS):
            f_pred = f(x[i], y_pred)
            y_corr = (
                    y[i - 1]
                    + (h / 24) * (
                            9 * f_pred
                            + 19 * f(x[i - 1], y[i - 1])
                            - 5 * f(x[i - 2], y[i - 2])
                            + 1 * f(x[i - 3], y[i - 3])
                    )
            )
            # проверяем сходимость по eps
            if abs(y_corr - y_pred) < eps:
                y_pred = y_corr
                break
            y_pred = y_corr
        else:
            raise Exception("Метод Адамса: не сошёлся корректор")

        y.append(y_pred)

    return y


def milne_method(f, x, y0, eps):
    """
    Метод Милна (предиктор–корректор, порядок 4).
    1) Старт: первые 4 точки — RK4.
    2) Predictor (Milne): явная формула с точек i-3..i-1.
    3) Corrector: итерация до |Δ|<eps.
    """
    n = len(x)
    y = fourth_order_runge_kutta_method(f, x[:4], y0)

    for i in range(4, n):
        h = x[i] - x[i - 1]
        # предсказание по Милну
        y_pred = (
                y[i - 4]
                + (4 * h / 3) * (
                        2 * f(x[i - 3], y[i - 3])
                        - 1 * f(x[i - 2], y[i - 2])
                        + 2 * f(x[i - 1], y[i - 1])
                )
        )

        # итеративная коррекция
        for it in range(MAX_ITERATIONS):
            f_pred = f(x[i], y_pred)
            y_corr = (
                    y[i - 2]
                    + (h / 3) * (
                            f(x[i - 2], y[i - 2])
                            + 4 * f(x[i - 1], y[i - 1])
                            + f_pred
                    )
            )
            if abs(y_corr - y_pred) < eps:
                y_pred = y_corr
                break
            y_pred = y_corr
        else:
            raise Exception("Метод Милна: не сошёлся корректор")

        y.append(y_pred)

    return y


ONE_STEP_METHODS = [
    euler_method,
    improved_euler_method,
    second_order_runge_kutta_method,
    fourth_order_runge_kutta_method
]

ONE_STEP_METHODS_NAMES = [
    'Метод Эйлера',
    'Модифицированный метод Эйлера',
    'Метод Рунге-Кутты 2-го порядка',
    'Метод Рунге-Кутты 4-го порядка',
]

RUNGE_P = [1, 2, 2, 4]

MULTY_STEP_METHODS = [
    adams_method,
    milne_method
]

MULTY_STEP_METHODS_NAMES = [
    'Метод Адамса',
    'Метод Милна'
]

EQUATIONS = [
    lambda x, y: y + (1 + x) * y ** 2,
    lambda x, y: x + y,
    lambda x, y: cos(x) - y
]

EQUATIONS_NAMES = [
    'y + (1 + x) * y^2',
    'x + y',
    'cos(x) - y'
]

EQUATIONS_SOLUTIONS = [
    lambda x, x0, y0: -exp(x) / (x * exp(x) - (x0 * exp(x0) * y0 + exp(x0)) / y0),
    lambda x, x0, y0: exp(x - x0) * (y0 + x0 + 1) - x - 1,
    lambda x, x0, y0: (y0 - sin(x0)) * exp(-(x - x0)) + sin(x)
]


def main():
    equation_id = choose_options('Выберите уравнение', EQUATIONS_NAMES) - 1
    equation = EQUATIONS[equation_id]
    solution = EQUATIONS_SOLUTIONS[equation_id]

    x0 = read_float('Введите первый элемент интервала')
    xn = read_float('Введите второй элемент интервала')

    if x0 > xn:
        x0, xn = xn, x0
        print('Значения x0 и xn были поменяны местами')

    n = read_positive_integer('Введите количество элементов в интервале')
    y0 = read_float('Введите y0')

    if not is_save_interval(x0, xn, lambda x_: solution(x_, x0, y0)):
        raise Exception('Решение определено не на всём интервале')

    eps = read_float('Введите точность')
    result_log = []

    for i in range(len(ONE_STEP_METHODS)):
        method = ONE_STEP_METHODS[i]
        method_name = ONE_STEP_METHODS_NAMES[i]
        p = RUNGE_P[i]

        try:
            result = solve_one_step(equation, method, p, x0, xn, n, y0, eps)
        except Exception as e:
            result_log.append(f'{method_name}\nОшибка: {e}')
            continue

        y_real = [solution(x_, x0, y0) for x_ in result.x]
        result_log.append(generate_result_log(method_name, result, y_real))

        solution_f = lambda x_: solution(x_, x0, y0)
        draw_plot(result.x, result.y, solution_f, method_name)

    for i in range(len(MULTY_STEP_METHODS)):
        method = MULTY_STEP_METHODS[i]
        method_name = MULTY_STEP_METHODS_NAMES[i]

        try:
            result = solve_multy_step(equation, method, solution, x0, xn, n, y0, eps)
        except Exception as e:
            result_log.append(f'{method_name}\nОшибка: {e}')
            continue

        y_real = [solution(x_, x0, y0) for x_ in result.x]
        result_log.append(generate_result_log(method_name, result, y_real))

        solution_f = lambda x_: solution(x_, x0, y0)
        draw_plot(result.x, result.y, solution_f, method_name)

    print_result(result_log)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
