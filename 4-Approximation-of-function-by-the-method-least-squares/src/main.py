import math
from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt


def pearson_correlation_coefficient(x, y, n):
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    r = (sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) /
         math.sqrt(sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y)))

    return r


def mean_squared_error(x, y, phi, n):
    return math.sqrt(sum(((phi(xi) - yi) ** 2 for xi, yi in zip(x, y))) / n)


def measure_of_deviation(x, y, phi):
    return sum(((phi(xi) - yi) ** 2 for xi, yi in zip(x, y)))


def coefficient_of_determination(x, y, phi, n):
    mean_phi = sum(phi(xi) for xi in x) / n
    return 1 - sum((yi - phi(xi)) ** 2 for xi, yi in zip(x, y)) / sum((yi - mean_phi) ** 2 for yi in y)


def linear_approximation(x, y, n):
    if n < 2:
        raise Exception('Должно быть минимум 2 точки')

    sx = sum(x)
    sxx = sum(xi ** 2 for xi in x)
    sy = sum(y)
    sxy = sum(xi * yi for xi, yi in zip(x, y))

    try:
        a, b = np.linalg.solve(
            [
                [sxx, sx],
                [sx, n]
            ],
            [sxy, sy]
        )
    except np.linalg.LinAlgError:
        raise Exception('Не удалось подобрать коэффициенты')

    phi = lambda x_: a * x_ + b

    return phi, (a, b)


def square_approximation(x, y, n):
    if n < 3:
        raise Exception('Должно быть минимум 3 точки')

    sx = sum(x)
    sxx = sum(xi ** 2 for xi in x)
    sxxx = sum(xi ** 3 for xi in x)
    sxxxx = sum(xi ** 4 for xi in x)
    sy = sum(y)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    sxxy = sum(xi * xi * yi for xi, yi in zip(x, y))

    try:
        a0, a1, a2 = np.linalg.solve(
            [
                [n, sx, sxx],
                [sx, sxx, sxxx],
                [sxx, sxxx, sxxxx]
            ],
            [sy, sxy, sxxy]
        )
    except np.linalg.LinAlgError:
        raise Exception('Не удалось подобрать коэффициенты')

    phi = lambda x_: a2 * x_ ** 2 + a1 * x_ + a0

    return phi, (a0, a1, a2)


def cubic_approximation(xs, ys, n):
    if n < 4:
        raise Exception('Должно быть минимум 4 точки')

    sx = sum(xs)
    sxx = sum(xi ** 2 for xi in xs)
    sxxx = sum(xi ** 3 for xi in xs)
    sxxxx = sum(xi ** 4 for xi in xs)
    sxxxxx = sum(xi ** 5 for xi in xs)
    sxxxxxx = sum(xi ** 6 for xi in xs)
    sy = sum(ys)
    sxy = sum(xi * yi for xi, yi in zip(xs, ys))
    sxxy = sum(xi * xi * yi for xi, yi in zip(xs, ys))
    sxxxy = sum(xi * xi * xi * yi for xi, yi in zip(xs, ys))

    try:
        a0, a1, a2, a3 = np.linalg.solve(
            [
                [n, sx, sxx, sxxx],
                [sx, sxx, sxxx, sxxxx],
                [sxx, sxxx, sxxxx, sxxxxx],
                [sxxx, sxxxx, sxxxxx, sxxxxxx]
            ],
            [sy, sxy, sxxy, sxxxy]
        )
    except np.linalg.LinAlgError:
        raise Exception('Не удалось подобрать коэффициенты')

    phi = lambda x_: a3 * x_ ** 3 + a2 * x_ ** 2 + a1 * x_ + a0

    return phi, (a0, a1, a2, a3)


def exponential_approximation(x, y, n):
    if n < 2:
        raise Exception('Должно быть минимум 2 точки')
    if min(y) <= 0:
        raise ValueError('Аппроксимация возможна только для наборов точек у которых y > 0')

    _, (a_, b_) = linear_approximation(x, np.log(y), n)

    a = a_
    b = np.exp(b_)

    phi = lambda x_: b * np.exp(a * x_)

    return phi, (a, b)


def logarithmic_approximation(x, y, n):
    if n < 2:
        raise Exception('Должно быть минимум 2 точки')
    if min(x) <= 0:
        raise ValueError('Аппроксимация возможна только для наборов точек у которых x > 0')

    _, (a_, b_) = linear_approximation(np.log(x), y, n)

    a = a_
    b = b_

    phi = lambda x_: a * np.log(np.clip(x_, 1e-10, None)) + b

    return phi, (a, b)


def power_approximation(x, y, n):
    if n < 2:
        raise Exception('Должно быть минимум 2 точки')
    if min(x) <= 0 or min(y) <= 0:
        raise ValueError('Аппроксимация возможна только для наборов точек у которых x > 0 и y > 0')

    _, (b_, a_) = linear_approximation(np.log(x), np.log(y), n)

    a = np.exp(a_)
    b = b_

    def phi(x_):
        if b < 0:
            x_ = np.where(x_ != 0, x_, 1e-10)
        if abs(b) < 1:
            x_ = np.clip(x_, 1e-10, None)

        return a * np.power(x_, b)

    return phi, (a, b)


class Writer:
    @abstractmethod
    def write(self, data):
        pass


class ConsoleWriter(Writer):
    def write(self, text):
        print(text)


class FileWriter(Writer):
    def __init__(self, path):
        self.file = open(path, 'w')

    def write(self, text):
        self.file.write(text + '\n')

    def close(self):
        self.file.close()


class Reader:
    @abstractmethod
    def read(self, text=None):
        pass


approximation_functions_coefficients_to_str = [
    lambda c: f'{round_(c[0], 3)}x + {round_(c[1], 3)}',
    lambda c: f'{round_(c[2], 3)}x^2 + {round_(c[1], 3)}x + {round_(c[0], 3)}',
    lambda c: f'{round_(c[3], 3)}x^3 + {round_(c[2], 3)}x^2 + {round_(c[1], 3)}x + {round_(c[0], 3)}',
    lambda c: f'{round_(c[1], 3)} * e^{round_(c[0], 3)}x',
    lambda c: f'{round_(c[0], 3)} * ln(x) + {round_(c[1], 3)}',
    lambda c: f'{round_(c[0], 3)} * x^{round_(c[1], 3)}'
]


def round_(n, precision):
    return "{:.{}f}".format(n, precision)


def print_result(result, writer):
    separator = '=' * 50
    line = f"\n\n{separator}\n\n".join(result)
    writer.write(f'\n{separator}\n\n{line}\n\n{separator}\n')


def create_reader():
    intput_mode = choose_options('Выберите способ ввода', IO_METHODS)
    reader = ConsoleReader()
    if intput_mode == 2:
        filename = read_filename('r')
        reader = FileReader(filename)
    return reader


def create_writer():
    output_mode = choose_options('Выберите способ вывода ответа', IO_METHODS)
    writer = ConsoleWriter()
    if output_mode == 2:
        filename = read_filename('w')
        writer = FileWriter(filename)
    return writer


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


class Writer:
    @abstractmethod
    def write(self, data):
        pass


APPROXIMATION_FUNCTIONS = [linear_approximation, square_approximation, cubic_approximation,
                           exponential_approximation, logarithmic_approximation, power_approximation]
APPROXIMATION_FUNCTIONS_NAMES = ['Линейная', 'Полиноминальная 2-й степени', 'Полиноминальная 3-й степени',
                                 'Экспоненциальная', 'Логарифмическая', 'Степенная']

IO_METHODS = ['Консоль', 'Файл']


def draw_plot(x, y, phis, names):
    x = np.array(x)
    y = np.array(y)

    x_min, x_max = min(x), max(x)
    x_margin = (x_max - x_min) * 0.1 if x_min != x_max else 1
    y_min, y_max = min(y), max(y)
    y_margin = (y_max - y_min) * 0.1 if y_min != y_max else 1

    plt.figure(figsize=(8, 6))

    plt.scatter(x, y, color='blue', label='Точки (x, y)')

    x_smooth = np.linspace(x_min - x_margin, x_max + x_margin, 1000)
    for i, phi in enumerate(phis):
        y_smooth = phi(x_smooth)
        plt.plot(x_smooth, y_smooth, label=names[i])

    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    reader = create_reader()
    if isinstance(reader, ConsoleReader):
        print('Вводите точки, по одной в строке. По окончании ввода введите q')
    x, y = read_points(reader)
    n = len(x)

    phis = []
    phis_names = []

    max_abs_r2 = 0
    best_approximation_function_index = None

    result = []

    for i in range(len(APPROXIMATION_FUNCTIONS)):
        f = APPROXIMATION_FUNCTIONS[i]
        name = APPROXIMATION_FUNCTIONS_NAMES[i]

        log = [f'Аппроксимирующая функция: {name}']

        try:
            phi, c = f(x, y, n)
        except Exception as e:
            log.append(f'ОШИБКА: {e}')
            result.append("\n".join(log))
            continue

        phis.append(phi)
        phis_names.append(name)

        phi_str = approximation_functions_coefficients_to_str[i](c)
        mse = mean_squared_error(x, y, phi, n)
        r2 = coefficient_of_determination(x, y, phi, n)
        s = measure_of_deviation(x, y, phi)

        if abs(r2) > max_abs_r2:
            max_abs_r2 = abs(r2)
            best_approximation_function_index = i

        log.append(f'Функция: φ(x) = {phi_str}')
        log.append(f'Среднеквадратичное отклонение: σ = {round_(mse, 3)}')
        log.append(f'Коэффициент детерминации: R² = {round_(r2, 3)}')
        log.append(f'Мера отклонения: S = {round_(s, 3)}')

        if f is linear_approximation:
            r = pearson_correlation_coefficient(x, y, n)
            log.append(f'Коэффициент кореляции Пирсона: r = {round_(r, 3)}')

        result.append("\n".join(log))

    writer = create_writer()

    print_result(result, writer)

    best_approximation_function_name = None
    if best_approximation_function_index is not None:
        best_approximation_function_name = APPROXIMATION_FUNCTIONS_NAMES[best_approximation_function_index]

    print(f'Лучшая аппроксимирующая функция: {best_approximation_function_name}')

    if n != 0: draw_plot(x, y, phis, phis_names)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
