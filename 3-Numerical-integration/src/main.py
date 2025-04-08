import math
from typing import List


# Класс для хранения результата вычислений: значение интеграла и количество итераций
class Result:
    def __init__(self, value, iterations):
        self.value = value
        self.iterations = iterations


# Обёртка над функцией: хранит саму функцию и её текстовое представление
class Function:
    def __init__(self, f, text):
        self.f = f
        self.text = text

    def compute(self, x):
        return self.f(x)

    def compute_or_none(self, x):
        try:
            return self.compute(x)
        except Exception:
            return None

    def __str__(self):
        return self.text


# Утилита округления до заданной точности
def round_(n: float, precision: int):
    return "{:.{}f}".format(n, precision)


# Вывод результата на экран
def print_result(result: Result):
    print(f'Найденное значение интеграла: {result.value}')
    print(f'Число разбиения интервала интегрирования для достижения требуемой точности: {result.iterations}')


# Выбор одного из нескольких вариантов (с пользовательским вводом)
def choose_options(message: str, options: List[str]) -> int:
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


# Ввод вещественного числа с проверкой
def read_float(message: str) -> float:
    value = None
    while value is None:
        try:
            value = float(input(f'{message}: '))
            break
        except:
            print('Значение должны быть целым или дробным числом!')
    return value


# Методы численного интегрирования:
# Метод правых прямоугольников
def rectangles_method_right(function: Function, a: float, b: float, n: int):
    h = (b - a) / n
    result = 0
    for i in range(1, n + 1):
        result += function.compute(a + i * h)
    return result * h


# Метод левых прямоугольников
def rectangles_method_left(function: Function, a: float, b: float, n: int):
    h = (b - a) / n
    result = 0
    for i in range(n):
        result += function.compute(a + i * h)
    return result * h


# Метод средних прямоугольников
def rectangles_method_middle(function: Function, a: float, b: float, n: int):
    h = (b - a) / n
    result = 0
    for i in range(1, n + 1):
        x_prev = a + (i - 1) * h
        x_i = a + i * h
        x_h = (x_prev + x_i) / 2
        result += function.compute(x_h)
    return result * h


# Метод трапеций
def trapezoid_method(function: Function, a: float, b: float, n: int):
    h = (b - a) / n
    result = (function.compute(a) + function.compute(b)) / 2
    for i in range(1, n):
        result += function.compute(a + i * h)
    return result * h


# Метод Симпсона (параболическое приближение)
def simpson_method(function: Function, a: float, b: float, n: int):
    h = (b - a) / n
    result = function.compute(a) + function.compute(b)
    for i in range(1, n):
        k = 2 if i % 2 == 0 else 4
        result += k * function.compute(a + i * h)
    return result * h / 3


# Итеративное уточнение результата до достижения точности (с использованием правила Рунге)
def calculate_integral(function: Function, a: float, b: float, eps: float, method, runge_k: int) -> Result:
    n = INIT_N
    result = method(function, a, b, n)
    delta = math.inf
    while delta > eps:
        if n >= MAX_N:
            raise Exception(f'Произведено разбиение на {MAX_N} отрезков, но ответ не найден')
        n *= 2
        new_result = method(function, a, b, n)
        delta = abs(new_result - result) / (2 ** runge_k - 1)
        result = new_result
    return Result(result, n)


# Поиск точек разрыва функции на отрезке [a, b]
def get_breaking_points(function: Function, a: float, b: float):
    n = math.ceil((b - a) / BREAKING_POINTS_ACCURACY)
    h = (b - a) / n
    breaking_points = []
    last_i = -2
    for i in range(n + 1):
        x = a + i * h
        if function.compute_or_none(x) is None:
            if i - 1 == last_i and i != n:
                raise Exception(
                    "Фунция может быть неопределена только в некоторых точках.\nНа выбранном отрезке существуют области неопредедённости.\nИнтегрирование невозможно")
            last_i = i
            breaking_points.append(x)
    return breaking_points


# Проверка на бесконечность
def is_inf(x):
    return abs(x) >= 1 / CONVERGENCE_EPS - 1 / BREAKING_POINTS_ACCURACY


# Проверка сходимости несобственного интеграла
def is_converges(function: Function, a: float, b: float, breaking_points: List[float]) -> bool:
    eps = CONVERGENCE_EPS
    breaking_points_tmp = breaking_points.copy()

    if a in breaking_points_tmp:
        breaking_points_tmp.remove(a)
        y = function.compute_or_none(a + eps)
        if y is None or is_inf(y):
            return False

    if b in breaking_points_tmp:
        breaking_points_tmp.remove(b)
        y = function.compute_or_none(b - eps)
        if y is None or is_inf(y):
            return False

    for p in breaking_points_tmp:
        y1 = function.compute_or_none(p - eps)
        y2 = function.compute_or_none(p + eps)
        if (y1 is None and y2 is None) or (is_inf(y1) and is_inf(y2) and y1 * y2 > 0):
            return False

    return True


# Вычисление несобственного интеграла по частям, с проверкой разрывов
def calculate_improper_integral(function: Function, a: float, b: float, eps: float, method: Function, runge_k: int,
                                breaking_points: List[float]) -> Result:
    conv_eps = CONVERGENCE_EPS
    result = 0
    iterations = 0

    if a not in breaking_points:
        b_ = breaking_points[0] - conv_eps
        y = function.compute_or_none(b_)
        if y is not None and not is_inf(y):
            result_ = calculate_integral(function, a, b_, eps, method, runge_k)
            result += result_.value
            iterations += result_.iterations

    if b not in breaking_points:
        a_ = breaking_points[-1] + conv_eps
        y = function.compute_or_none(a_)
        if y is not None and not is_inf(y):
            result_ = calculate_integral(function, a_, b, eps, method, runge_k)
            result += result_.value
            iterations += result_.iterations

    for i in range(1, len(breaking_points)):
        a_ = breaking_points[i] - conv_eps
        b_ = breaking_points[i - 1] + conv_eps
        y_a_ = function.compute_or_none(a_)
        y_b_ = function.compute_or_none(b_)

        if y_a_ is not None and y_b_ is not None and not (is_inf(y_a_) and is_inf(y_b_) and y_a_ * y_b_ > 0):
            result_ = calculate_integral(function, a_, b_, eps, method, runge_k)
            result += result_.value
            iterations += result_.iterations

    return Result(result, iterations)


# Конфигурация
INIT_N = 4
MAX_N = 1_000_000
BREAKING_POINTS_ACCURACY = 1e-4
CONVERGENCE_EPS = 1e-9
METHODS_STRS = ['Метод прямоугольников (левый)', 'Метод прямоугольников (правый)', 'Метод прямоугольников (средний)',
                'Метод трапеций', 'Метод Симпсона']
METHODS = [rectangles_method_left, rectangles_method_right, rectangles_method_middle, trapezoid_method, simpson_method]
METHODS_RUNGE_K = [1, 1, 2, 2, 4]  # Порядок точности методов
FUNCTIONS = [
    Function(lambda x: x ** 2, 'x^2'),
    Function(lambda x: math.sin(x), 'sin(x)'),
    Function(lambda x: x ** 3 - 3 * x ** 2 + 7 * x - 10, 'x^3 - 3x^2 + 7x - 10'),
    Function(lambda x: 5, '5'),
    Function(lambda x: 1 / math.sqrt(x), '1 / sqrt(x)'),
    Function(lambda x: 1 / (1 - x), '1 / (1 - x)'),
    Function(lambda x: 1 / x, '1 / x'),
    Function(lambda x: 1 / x ** 2, '1 / x^2')
]


def main():
    # Выбор функции
    function_id = choose_options('Выберите функцию для интегрирования', FUNCTIONS) - 1
    function = FUNCTIONS[function_id]

    # Ввод границ интегрирования
    a = read_float('Введите нижний предел интегрирования')
    b = read_float('Введите верхний предел интегрирования')

    is_inv = False
    if b < a:
        a, b = b, a
        is_inv = True  # Если порядок границ нарушен, меняем и запоминаем

    # Проверка на наличие разрывов
    is_improper_integral = False
    breaking_points = get_breaking_points(function, a, b)
    if len(breaking_points) != 0:
        print(f'Функция терпит разрыв в точках: {breaking_points}')
        is_improper_integral = True

    if is_improper_integral and not is_converges(function, a, b, breaking_points):
        raise Exception('Интеграл расходится')

    # Выбор метода
    method_id = choose_options('Выберите метод для интегрирования', METHODS_STRS) - 1
    method = METHODS[method_id]
    runge_k = METHODS_RUNGE_K[method_id]

    # Ввод точности
    eps = read_float('Введите точность')

    # Вычисление интеграла
    if not is_improper_integral:
        result = calculate_integral(function, a, b, eps, method, runge_k)
    else:
        result = calculate_improper_integral(function, a, b, eps, method, runge_k, breaking_points)

    # Учёт изменения порядка границ
    if is_inv:
        result.value *= -1

    print_result(result)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
