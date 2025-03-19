from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from scipy.differentiate import derivative
from scipy.optimize import root
from tabulate import tabulate


# Reader
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


class Writer:
    @abstractmethod
    def write(self, data):
        pass


# DTO
class Result:
    def __init__(self, x, iterations, log=None):
        self.x = x
        self.iterations = iterations
        self.log = log


# Multi Equation
class MultiEquation:
    def __init__(self, f, text):
        self.f = f
        self.text = text

    def __str__(self):
        return self.text

    def partial_derivative(self, x, i):
        g = lambda _x: self.f(x[:i] + [_x] + x[i + 1:])
        return derivative(g, x[i]).df


# Simple Equation
class SimpleEquation:
    def __init__(self, f, text):
        self.f = f
        self.text = text

    def __str__(self):
        return self.text

    def fst_derivative(self, x):
        return derivative(self.f, x).df

    def snd_derivative(self, x):
        return derivative(self.fst_derivative, x).df

    def is_single_root_exist(self, left, right):
        if self.f(left) * self.f(right) > 0:
            return False
        for x in np.linspace(left, right, int((right - left) * 10)):
            if self.fst_derivative(left) * self.fst_derivative(x) < 0:
                return False
        return True


# System Of Equations
class SystemOfEquations:
    def __init__(self, equations):
        self.equations = equations
        self.n = len(equations)

    def __str__(self):
        return '{' + ''.join(f'{str(eq)}; ' for eq in self.equations) + '}'

    def partial_derivative(self, x, i, j):
        return self.equations[i].partial_derivative(x, j)

    def get_jacobi(self, x):
        jcb = [[] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                jcb[i].append(self.partial_derivative(x, i, j))
        return jcb

    def get_value(self, x):
        v = []
        for i in self.equations:
            v.append(i.f(x))
        return v


# Writer
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


# UTIL
def create_reader():
    intput_mode = choose_options('Выберите способ ввода границ интервала и точности', IO_METHODS)
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


def print_log(log, writer, log_decimals):
    header = list(log[0].keys())
    data = [
        [f'{v:.{log_decimals}f}' if isinstance(v, (int, float)) else
         [f'{num:.{log_decimals}f}' for num in v] if isinstance(v, list) else str(v)
         for v in item.values()]
        for item in log
    ]
    writer.write(tabulate(data, header, tablefmt='pretty', showindex=True))


def print_result(result, real_root, writer, log_decimals):
    writer.write(f'Найденный корень: {result.x}')
    writer.write(f'Истинный корень: {real_root}')
    writer.write(f'Потребовалось итераций: {result.iterations}')
    writer.write('Лог решения:')
    print_log(result.log, writer, log_decimals)


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


def read_root_limits(reader):
    left = None
    right = None
    while left is None or right is None:
        message = ''
        try:
            left, right = map(float, reader.read('Введите нижнюю и верхнюю границу диапазона корней: ').split())
            if left > right:
                message = 'Левая граница должна быть меньше левой!'
                left = None
                right = None
            else:
                break
        except:
            message = 'Значения должны быть числами!'
        if isinstance(reader, FileReader):
            raise Exception(message)
        else:
            print(message)
    return left, right


def read_eps(reader):
    eps = None
    while eps is None:
        try:
            eps = float(reader.read('Введите точность: '))
            break
        except:
            message = 'Значение должно быть числом'
        if isinstance(reader, FileReader):
            raise Exception(message)
        else:
            print(message)
    return eps


def read_initial_point(reader, n):
    point = None
    while point is None:
        message = ''
        try:
            point = list(map(float, reader.read(f'Введите {n} координат начального приближения: ').split()))
            if len(point) != n:
                print(f'Введите {n} чисел!')
                point = None
            else:
                break
        except:
            message = 'Значения должны быть числами!'
        if isinstance(reader, FileReader):
            raise Exception(message)
        else:
            print(message)
    return point


# Equation Methods
def chord_method(equation, a, b, eps):
    f = equation.f
    iterations = 0
    log = []

    x = (a * f(b) - b * f(a)) / (f(b) - f(a))

    while True:
        if iterations == MAX_ITERATIONS:
            raise Exception(f'Произведено {MAX_ITERATIONS} итераций, но решение не найдено.')
        iterations += 1

        if f(a) * f(x) <= 0:
            b = x
        else:
            a = x

        next_x = (a * f(b) - b * f(a)) / (f(b) - f(a))
        delta = abs(next_x - x)

        log.append({
            'a': a,
            'b': b,
            'x': x,
            'f(a)': f(a),
            'f(b)': f(b),
            'f(x)': f(x),
            'delta': delta})

        if delta < eps:
            break

        x = next_x

    return Result(x, iterations, log)


def secant_method(equation, a, b, eps):
    f = equation.f
    f__ = equation.snd_derivative
    iterations = 0
    log = []

    if f__(a) * f__(b) < 0:
        raise Exception(
            'Условия сходимости метода секущих не выполнены! Вторая производная не сохраняет знак на выбранном отрезке')

    x0 = a
    if f(a) * f__(a) > 0:
        x0 = a
    if f(b) * f__(b) > 0:
        x0 = b

    x1 = x0 + eps

    while True:
        if iterations == MAX_ITERATIONS:
            raise Exception(f'Произведено {MAX_ITERATIONS} итераций, но решение не найдено.')
        iterations += 1

        x = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        delta = abs(x - x1)

        log.append({
            'x_{i-1}': x0,
            'x_i': x1,
            'x_{i+1}': x,
            'f(x_{i+1})': f(x),
            'delta': delta
        })

        if delta < eps:
            break

        x0 = x1
        x1 = x

    return Result(x, iterations, log)


def simple_iteration_method(equation, a, b, eps):
    f = equation.f
    f_ = equation.fst_derivative
    iterations = 0
    log = []

    max_derivative = max(abs(f_(a)), abs(f_(b)))
    _lambda = 1 / max_derivative
    if f_(a) > 0: _lambda *= -1

    phi = lambda x: x + _lambda * f(x)

    phi_ = lambda x: derivative(phi, x).df
    q = np.max(abs(phi_(np.linspace(a, b, int(1 / eps)))))
    if q > 1:
        raise Exception(f'Метод не сходится так как значение q >= 1')

    prev_x = a
    while True:
        if iterations == MAX_ITERATIONS:
            raise Exception(f'Произведено {MAX_ITERATIONS} итераций, но решение не найдено.')
        iterations += 1

        x = phi(prev_x)
        delta = abs(x - prev_x)

        log.append({
            'x_i': prev_x,
            'x_{i+1}': x,
            'phi(x_{i+1})': phi(x),
            'f(x_{i+1})': f(x),
            'delta': delta
        })

        if delta <= eps:
            break

        prev_x = x

    return Result(x, iterations, log)


# System methods
def newton_method(system, x0, eps):
    x = x0
    iterations = 0
    log = []
    while True:
        if iterations == MAX_ITERATIONS:
            raise Exception(f'Произведено {MAX_ITERATIONS} итераций, но решение не найдено.')
        iterations += 1

        jcb = system.get_jacobi(x)
        b = system.get_value(x)
        try:
            dx = np.linalg.solve(np.array(jcb), -1 * np.array(b))
        except np.linalg.LinAlgError:
            raise Exception('Не удалось применить метод, промежуточная система не имеет решений!')
        nx = x + dx

        log.append({
            'x_i': x,
            'x_{i+1}': nx.tolist(),
            'dx': dx.tolist()
        })

        if np.max(np.abs(nx - x)) <= eps:
            break
        x = nx.tolist()

    return Result(x, iterations, log)


# Drawers
def draw_equation(x0, left, right, equation):
    side_step = abs(right - left) * 0.15
    l = left - side_step
    r = right + side_step

    plt.figure(figsize=(10, 10))

    x = np.linspace(l, r, 1000)
    y = equation.f(x)
    plt.plot(x, y, label=f'f(x)', color='blue')

    y0 = equation.f(x0)
    plt.scatter([x0], [y0], label=f'({_round(x0, 3)}; {_round(y0, 3)})', color='red', s=50)

    x_l = left
    y_l = equation.f(x_l)
    plt.vlines(x_l, 0, y_l, colors='black', linestyles='--')
    plt.scatter([x_l], [y_l], color='black', s=50)

    x_r = right
    y_r = equation.f(x_r)
    plt.vlines(x_r, 0, y_r, colors='black', linestyles='--')
    plt.scatter([x_r], [y_r], color='black', s=50)

    plt.axhline(0, color='black')

    plt.title(f'График функции f(x)={equation.text}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.xlim(l, r)
    plt.show()


def draw_system(x0, point, system):
    if system.n > 2: return
    r = np.sqrt(max(x0[0] ** 2 + x0[1] ** 2, point[0] ** 2 + point[1] ** 2)) * 1.5
    x_min, x_max = -r, r
    y_min, y_max = -r, r

    plt.figure(figsize=(10, 10))

    x = np.linspace(x_min, x_max, 1000)
    y = np.linspace(y_min, y_max, 1000)
    x, y = np.meshgrid(x, y)
    f = system.equations[0].f([x, y])
    g = system.equations[1].f([x, y])
    plt.contour(x, y, f, levels=[0], colors='blue')
    plt.contour(x, y, g, levels=[0], colors='green')

    plt.scatter([x0[0]], [x0[1]], label=f'({_round(x0[0], 3)}; {_round(x0[1], 3)})', color='red', s=50)
    plt.scatter([point[0]], [point[1]], label=f'({_round(point[0], 3)}; {_round(point[1], 3)})', color='black', s=50)

    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    plt.title(f'График функции f(x)={system.equations[0].text}')
    plt.title(f'График функции g(x)={system.equations[1].text}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()


# IO util
def _round(n, precision):
    return "{:.{}f}".format(n, precision)


# CONFIG
LOG_DECIMALS = 2
MAX_ITERATIONS = 1000
MODES = ['Нелинейное уравнение', 'Система нелинейных уравнений']
EQ_METHODS_STRS = ['Метод хорд', 'Метод секущих', 'Метод простых итераций']
EQ_METHODS = [chord_method, secant_method, simple_iteration_method]
SYS_METHODS_STRS = ['Метод Ньютона']
SYS_METHODS = [newton_method]
IO_METHODS = ['Консоль', 'Файл']
EQUATIONS = [
    SimpleEquation(lambda x: x ** 3 - x + 4, 'x^3 - x + 4'),
    SimpleEquation(lambda x: x ** 3 - x ** 2 - 25 * x + 2, 'x^3 - x^2 - 25x + 2'),
    SimpleEquation(lambda x: np.atan(x), 'arctg(x)')
]
SYSTEMS = [
    SystemOfEquations([
        MultiEquation(lambda x_: x_[0] ** 2 + x_[1] ** 2 - 4, 'x^2 + y^2 = 4'),
        MultiEquation(lambda x_: -3 * x_[0] ** 2 + x_[1], 'y = 3x^2')
    ]),
    SystemOfEquations([
        MultiEquation(lambda x_: x_[0] ** 2 + x_[1] ** 2 - 4, 'x^2 + y^2 = 4'),
        MultiEquation(lambda x_: x_[1] - np.sin(x_[0]), 'y = sin(x)')
    ]),
    SystemOfEquations([
        MultiEquation(lambda x_: x_[0] ** 2 + x_[1] ** 2 - 6, 'x^2 + y^2 = 6'),
        MultiEquation(lambda x_: x_[1] - np.tan(x_[0]), 'y = tg(x)')
    ])
]


def main():
    mode = choose_options('Выберите что будете решать', MODES)

    if mode == 1:
        equation_id = choose_options('Выберите уравнение', EQUATIONS) - 1
        method_id = choose_options('Выберите метод', EQ_METHODS_STRS) - 1
        method = EQ_METHODS[method_id]
        equation = EQUATIONS[equation_id]
        reader = create_reader()
        a, b = read_root_limits(reader)

        if not equation.is_single_root_exist(a, b):
            raise Exception('На выбранном отрезке нет корней либо их больше одного')

        eps = read_eps(reader)
        result = method(equation, a, b, eps)
        writer = create_writer()
        real_root = root(equation.f, (a + b) / 2).x[0]
        print_result(result, real_root, writer, LOG_DECIMALS)
        draw_equation(result.x, a, b, equation)

    if mode == 2:
        system_id = choose_options('Выберите систему', SYSTEMS) - 1
        method_id = choose_options('Выберите метод', SYS_METHODS_STRS) - 1
        reader = create_reader()
        initial_point = read_initial_point(reader, SYSTEMS[system_id].n)
        eps = read_eps(reader)
        system = SYSTEMS[system_id]
        method = SYS_METHODS[method_id]
        result = method(system, initial_point, eps)
        writer = create_writer()
        real_root = root(system.get_value, initial_point).x.tolist()
        print_result(result, real_root, writer, LOG_DECIMALS)
        draw_system(result.x, initial_point, system)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
