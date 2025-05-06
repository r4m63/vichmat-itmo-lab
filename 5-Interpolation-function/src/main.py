import numpy as np
from matplotlib import pyplot as plt
from math import sin, sqrt
from abc import abstractmethod


def lagrange_polynomial(x, y):
    n = len(x)

    def p(x_):
        result = 0
        for k in range(n):
            nominator = 1
            denominator = 1
            for i in range(n):
                if i == k:
                    continue
                nominator *= x_ - x[i]
                denominator *= x[k] - x[i]
            result += y[k] * (nominator / denominator)
        return result

    return p


def newton_divided_difference_polynomial(x, y):
    n = len(x)
    diffs = calculate_divided_differences(x, y)

    def p(x_):
        result = y[0]
        for k in range(1, n):
            d = diffs[k]
            for i in range(0, k):
                d *= (x_ - x[i])
            result += d
        return result

    return p


def _first_gauss_polynomial(x, y):
    n = len(x)
    h = x[1] - x[0]
    alpha_ind = n // 2
    diffs = calculate_finite_difference_table(y)

    def p(x_):
        t = (x_ - x[alpha_ind]) / h
        result = 0

        for k in range(n):
            m = (k + 1) // 2

            nominator = 1
            for j in range(-(m - 1), m):
                nominator *= t + j
            if k == 2 * m and m != 0: nominator *= t - m

            factorial = 1
            for j in range(1, k + 1):
                factorial *= j

            if k == 2 * m:
                result += diffs[alpha_ind - m][k] * (nominator / factorial)
            else:
                result += diffs[alpha_ind - (m - 1)][k] * (nominator / factorial)
        return result

    return p


def _second_gauss_polynomial(x, y):
    n = len(x)
    h = x[1] - x[0]
    alpha_ind = n // 2
    diffs = calculate_finite_difference_table(y)

    def p(x_):
        t = (x_ - x[alpha_ind]) / h
        result = 0

        for k in range(n):
            m = (k + 1) // 2

            nominator = 1
            for j in range(-(m - 1), m):
                nominator *= t + j
            if k == 2 * m and m != 0: nominator *= t + m

            factorial = 1
            for j in range(1, k + 1):
                factorial *= j

            result += diffs[alpha_ind - m][k] * (nominator / factorial)
        return result

    return p


def gauss_polynomial(x, y):
    if not is_finite_difference(x):
        raise Exception('Значения X должны иметь фиксированный шаг!')

    n = len(x)
    alpha_ind = n // 2

    p1 = _first_gauss_polynomial(x, y)
    p2 = _second_gauss_polynomial(x, y)

    p = lambda x_: p1(x_) if x_ > x[alpha_ind] else p2(x_)

    return p


def stirling_polynomial(x, y):
    if len(x) % 2 != 1:
        raise Exception('Число узлов должно быть нечетным')
    if not is_finite_difference(x):
        raise Exception('Значения X должны иметь фиксированный шаг!')

    p1 = _first_gauss_polynomial(x, y)
    p2 = _second_gauss_polynomial(x, y)

    p = lambda x_: (p1(x_) + p2(x_)) / 2

    return p


def bessel_polynomial(x, y):
    if len(x) % 2 != 0:
        raise Exception('Число узлов должно быть четным')
    if not is_finite_difference(x):
        raise Exception('Значения X должны иметь фиксированный шаг!')

    n = len(x)
    diffs = calculate_finite_difference_table(y)
    h = x[1] - x[0]
    alpha_ind = n // 2

    def p(x_):
        t = (x_ - x[alpha_ind]) / h
        result = 0

        for k in range(n):
            m = (k + 1) // 2

            nominator = 1
            for j in range(-m, m):
                nominator *= t + j
            if k == 2 * m - 1: nominator *= t - 0.5

            factorial = 1
            for j in range(1, k + 1):
                factorial *= j

            if k == 2 * m:
                result += ((diffs[alpha_ind - m][k] + diffs[alpha_ind - (m - 1)][k]) / 2) * (nominator / factorial)
            else:
                result += diffs[alpha_ind - m][k] * (nominator / factorial)
        return result

    return p

def calculate_finite_difference_table(y):
    n = len(y)
    delta_y = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        delta_y[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            delta_y[i][j] = delta_y[i + 1][j - 1] - delta_y[i][j - 1]
    return delta_y


def calculate_divided_differences(x, y):
    n = len(y)
    k = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        k[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            k[i][j] = (k[i + 1][j - 1] - k[i][j - 1]) / (x[i + j] - x[i])

    return k[0]


def is_finite_difference(x):
    n = len(x)
    h = x[1] - x[0]
    for i in range(1, n):
        if abs((x[i] - x[i - 1]) - h) > 1e-6:
            return False
    return True

# reader
class Reader:
    @abstractmethod
    def read(self, text=None):
        pass


class ConsoleReader(Reader):
    def read(self, text=None):
        return input() if text is None else input(text)


class FileReader(Reader):
    def __init__(self, path):
        self.file = open(path, 'r')

    def read(self, text=None):
        return self.file.readline()

    def close(self):
        self.file.close()

# util
def round_(n, precision):
    return "{:.{}f}".format(n, precision)


def print_finite_difference_table(table):
    n = len(table)
    for i in range(n):
        print(*[round_(table[i][j], 4) if i + j < n else '' for j in range(n)], sep='\t')


def print_result(result):
    separator = '=' * 50
    line = f"\n\n{separator}\n\n".join(result)
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


def read_filename(mode):
    filename = None
    while filename is None:
        filename = input('Введите имя файла: ').strip()
        try:
            open(filename, mode).close()
        except:
            filename = None
            print('Не удалось найти файл!')
    return filename


def read_points(reader):
    if isinstance(reader, ConsoleReader):
        print('Вводите точки, по одной в строке. По окончании ввода введите q')
    x = []
    y = []
    while True:
        try:
            s = reader.read()

            if s == 'q' or s == '' and isinstance(reader, FileReader):
                break

            xi, yi = list(map(float, s.split()))
            x.append(xi)
            y.append(yi)
        except:
            message = 'Некорректный ввод'
            if isinstance(reader, FileReader):
                raise Exception(message)
            else:
                print(message)

    return x, y


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


def read_point(message, function=None):
    if function is None:
        return read_float(message)

    value = None
    while value is None:
        try:
            value = read_float(message)
            function(value)
        except:
            print('Значение функции не определено в данной точке. Попробуйте снова')
            value = None
    return value

# constants
INTERPOLATION_FUNCTIONS = [
    lagrange_polynomial,
    newton_divided_difference_polynomial,
    gauss_polynomial,
    stirling_polynomial,
    bessel_polynomial
]
INTERPOLATION_FUNCTIONS_NAMES = [
    'Интерполяционный многочлен Лагранжа',
    'Интерполяционный многочлен Ньютона с разделенными разностями',
    'Интерполяционный многочлен Гаусса',
    'Интерполяционный многочлен Стирлинга',
    'Интерполяционный многочлен Бесселя'
]

DATA_SOURCES = ['Консоль', 'Файл', 'Функция']

FUNCTIONS = [
    lambda x: x ** 2,
    lambda x: x ** 3,
    lambda x: x ** 5,
    lambda x: sin(x),
    lambda x: sqrt(x)
]

FUNCTIONS_NAMES = [
    'x^2',
    'x^3',
    'x^5',
    'sin(x)',
    'sqrt(x)'
]


# data loaders

def validate_data(x, y):
    if len(set(x)) != len(x):
        raise Exception('Узлы интерполяции не должны совпадать!')
    elif x != sorted(x):
        raise Exception('Узлы интерполяции должны быть отсортированы по значению X!')


def load_from_console():
    reader = ConsoleReader()
    x, y = read_points(reader)
    validate_data(x, y)
    x0 = read_point('Введите точку интерполяции')
    return x0, x, y


def load_from_file():
    filename = read_filename('r')
    reader = FileReader(filename)
    x, y = read_points(reader)
    validate_data(x, y)
    x0 = read_point('Введите точку интерполяции')
    return x0, x, y


def load_from_function():
    function_id = choose_options('Выберите функцию', FUNCTIONS_NAMES) - 1
    function = FUNCTIONS[function_id]

    l = read_point('Введите левую границу исследуемого интервала', function)
    r = read_point('Введите правую границу исследуемого интервала', function)

    if l == r:
        raise Exception('Задан интервал нулевой длинны!')

    if l > r:
        l, r = r, l
        print('Левая граница больше правой. Границы были поменяны местами')

    n = read_positive_integer('Введите число узлов')

    h = (r - l) / (n - 1)
    x = [l + h * i for i in range(n)]
    y = list(map(function, x))

    x0 = read_point('Введите точку интерполяции', function)

    return x0, x, y

# drawer

def draw_plot(x, y, x0, polynomes, names):
    x = np.array(x)
    y = np.array(y)

    y0 = [p(x0) for p in polynomes]
    x0 = [x0 for _ in polynomes]

    x_min, x_max = min([*x, *x0]), max([*x, *x0])
    x_margin = (x_max - x_min) * 0.1 if x_min != x_max else 1
    y_min, y_max = min([*y, *y0]), max([*y, *y0])
    y_margin = (y_max - y_min) * 0.1 if y_min != y_max else 1

    plt.figure(figsize=(8, 6))

    plt.scatter(x, y, color='blue', label='Точки (x, y)')

    x_smooth = np.linspace(x_min - x_margin, x_max + x_margin, 1000)
    for i, p in enumerate(polynomes):
        y_smooth = np.array(list(map(p, x_smooth)))
        plt.plot(x_smooth, y_smooth, label=names[i])

    plt.scatter(x0, y0, color='red', label='Исследуемая (x, y)')

    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()



def main():
    mode = choose_options('Выберите способ задания функции', DATA_SOURCES)

    if mode == 1:
        x0, x, y = load_from_console()
    elif mode == 2:
        x0, x, y = load_from_file()
    else:
        x0, x, y = load_from_function()

    finite_difference_table = calculate_finite_difference_table(y)

    print('Таблица конечных разностей:')
    print_finite_difference_table(finite_difference_table)

    interpolation_polynomes = []
    interpolation_polynomes_names = []

    result = []

    for i in range(len(INTERPOLATION_FUNCTIONS)):
        f = INTERPOLATION_FUNCTIONS[i]
        name = INTERPOLATION_FUNCTIONS_NAMES[i]

        try:
            p = f(x, y)
        except Exception as e:
            result.append(f'Интерполирующая функция: {name}\nОШИБКА: {e}')
            continue

        result.append(f'Интерполирующая функция: {name}\nP({x0}) = {p(x0)}')

        if f not in [bessel_polynomial, stirling_polynomial]:
            interpolation_polynomes.append(p)
            interpolation_polynomes_names.append(name)

    print_result(result)

    draw_plot(x, y, x0, interpolation_polynomes, interpolation_polynomes_names)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)