import math
from abc import abstractmethod
import numpy as np
from matplotlib import pyplot as plt


# Вычисление коэффициента корреляции Пирсона для двух наборов данных x и y
def pearson_correlation_coefficient(x, y, n):
    mean_x = sum(x) / n  # Находим среднее значение x
    mean_y = sum(y) / n  # Находим среднее значение y

    # Вычисляем числитель и знаменатель формулы коэффициента Пирсона
    r = (sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) /
         math.sqrt(sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y)))

    return r  # Возвращаем коэффициент корреляции r


# Вычисление среднеквадратичного отклонения между аппроксимацией phi(x) и реальными данными y
def mean_squared_error(x, y, phi, n):
    # Для каждой точки вычисляем квадрат разности phi(xi) и yi, суммируем и делим на n, затем берём корень
    return math.sqrt(sum(((phi(xi) - yi) ** 2 for xi, yi in zip(x, y))) / n)


# Вычисление меры отклонения: сумма квадратов разностей между phi(x) и y
def measure_of_deviation(x, y, phi):
    # Просто суммируем квадраты ошибок без деления на количество точек
    return sum(((phi(xi) - yi) ** 2 for xi, yi in zip(x, y)))


# Вычисление коэффициента детерминации R²
def coefficient_of_determination(x, y, phi, n):
    mean_phi = sum(phi(xi) for xi in x) / n  # Сначала считаем среднее значение функции phi(x)

    # Формула R²: 1 - (сумма квадратов ошибок)/(сумма квадратов отклонений относительно среднего phi)
    return 1 - sum((yi - phi(xi)) ** 2 for xi, yi in zip(x, y)) / sum((yi - mean_phi) ** 2 for yi in y)


# ===== ФУНКЦИИ АПРОКСИМАЦИИ =====

# Линейная аппроксимация методом наименьших квадратов
def linear_approximation(x, y, n):
    if n < 2:  # Проверка: для прямой нужно минимум 2 точки
        raise Exception('Должно быть минимум 2 точки')

    # Вычисляем суммы, которые входят в нормальные уравнения
    sx = sum(x)  # сумма всех xi
    sxx = sum(xi ** 2 for xi in x)  # сумма всех xi^2
    sy = sum(y)  # сумма всех yi
    sxy = sum(xi * yi for xi, yi in zip(x, y))  # сумма всех xi*yi

    try:
        # Решаем систему нормальных уравнений для нахождения коэффициентов a и b
        a, b = np.linalg.solve(
            [
                [sxx, sx],  # коэффициенты при a и b для уравнения 1
                [sx, n]  # коэффициенты при a и b для уравнения 2
            ],
            [sxy, sy]  # правая часть уравнений
        )
    except np.linalg.LinAlgError:
        raise Exception('Не удалось подобрать коэффициенты')  # Обработка ошибки в случае вырожденной системы

    # phi(x) — найденная линейная функция: a * x + b
    phi = lambda x_: a * x_ + b

    return phi, (a, b)  # Возвращаем функцию и её коэффициенты


# Квадратичная аппроксимация методом наименьших квадратов
def square_approximation(x, y, n):
    if n < 3:  # Для квадратичной функции требуется минимум 3 точки
        raise Exception('Должно быть минимум 3 точки')

    # Вычисляем суммы для системы нормальных уравнений 3x3
    sx = sum(x)  # сумма xi
    sxx = sum(xi ** 2 for xi in x)  # сумма xi^2
    sxxx = sum(xi ** 3 for xi in x)  # сумма xi^3
    sxxxx = sum(xi ** 4 for xi in x)  # сумма xi^4
    sy = sum(y)  # сумма yi
    sxy = sum(xi * yi for xi, yi in zip(x, y))  # сумма xi*yi
    sxxy = sum(xi * xi * yi for xi, yi in zip(x, y))  # сумма xi^2*yi

    try:
        # Решаем систему уравнений для нахождения коэффициентов квадратичной функции
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

    # phi(x) — квадратичная функция
    phi = lambda x_: a2 * x_ ** 2 + a1 * x_ + a0

    return phi, (a0, a1, a2)  # Возвращаем функцию и коэффициенты


# Кубическая аппроксимация методом наименьших квадратов
def cubic_approximation(xs, ys, n):
    if n < 4:  # Для кубической функции нужно минимум 4 точки
        raise Exception('Должно быть минимум 4 точки')

    # Вычисляем суммы для нормальных уравнений 4x4
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
        # Решение системы для коэффициентов кубической функции
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

    # phi(x) — кубическая функция
    phi = lambda x_: a3 * x_ ** 3 + a2 * x_ ** 2 + a1 * x_ + a0

    return phi, (a0, a1, a2, a3)


# Экспоненциальная аппроксимация вида y = b * e^(a*x)
def exponential_approximation(x, y, n):
    if n < 2:
        raise Exception('Должно быть минимум 2 точки')
    if min(y) <= 0:
        raise ValueError('Аппроксимация возможна только для наборов точек у которых y > 0')

    # Логарифмируем y для приведения задачи к линейной
    _, (a_, b_) = linear_approximation(x, np.log(y), n)

    a = a_  # Параметр наклона экспоненты
    b = np.exp(b_)  # Преобразуем константу обратно из логарифма

    # phi(x) — экспоненциальная функция
    phi = lambda x_: b * np.exp(a * x_)

    return phi, (a, b)


# Логарифмическая аппроксимация вида y = a * ln(x) + b
def logarithmic_approximation(x, y, n):
    if n < 2:
        raise Exception('Должно быть минимум 2 точки')
    if min(x) <= 0:
        raise ValueError('Аппроксимация возможна только для наборов точек у которых x > 0')

    # Логарифмируем x для приведения задачи к линейной
    _, (a_, b_) = linear_approximation(np.log(x), y, n)

    a = a_  # Коэффициент при ln(x)
    b = b_  # Свободный член

    # phi(x) — логарифмическая функция
    phi = lambda x_: a * np.log(np.clip(x_, 1e-10, None)) + b  # Защита от log(0)

    return phi, (a, b)


# Степенная аппроксимация вида y = a * x^b
def power_approximation(x, y, n):
    if n < 2:
        raise Exception('Должно быть минимум 2 точки')
    if min(x) <= 0 or min(y) <= 0:
        raise ValueError('Аппроксимация возможна только для наборов точек у которых x > 0 и y > 0')

    # Логарифмируем обе переменные для приведения к линейной задаче
    _, (b_, a_) = linear_approximation(np.log(x), np.log(y), n)

    a = np.exp(a_)  # Параметр масштаба (a)
    b = b_  # Показатель степени (b)

    def phi(x_):
        # Защита от нулей при вычислениях
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


# Список функций для преобразования коэффициентов разных аппроксимаций в строку
approximation_functions_coefficients_to_str = [
    # 1. Линейная аппроксимация: phi(x) = a*x + b
    lambda c: f'{round_(c[0], 3)}x + {round_(c[1], 3)}',

    # 2. Квадратичная аппроксимация: phi(x) = a2*x^2 + a1*x + a0
    lambda c: f'{round_(c[2], 3)}x^2 + {round_(c[1], 3)}x + {round_(c[0], 3)}',

    # 3. Кубическая аппроксимация: phi(x) = a3*x^3 + a2*x^2 + a1*x + a0
    lambda c: f'{round_(c[3], 3)}x^3 + {round_(c[2], 3)}x^2 + {round_(c[1], 3)}x + {round_(c[0], 3)}',

    # 4. Экспоненциальная аппроксимация: phi(x) = b * e^(a*x)
    lambda c: f'{round_(c[1], 3)} * e^{round_(c[0], 3)}x',

    # 5. Логарифмическая аппроксимация: phi(x) = a * ln(x) + b
    lambda c: f'{round_(c[0], 3)} * ln(x) + {round_(c[1], 3)}',

    # 6. Степенная аппроксимация: phi(x) = a * x^b
    lambda c: f'{round_(c[0], 3)} * x^{round_(c[1], 3)}'
]


# Функция округления чисел до заданной точности
def round_(n, precision):
    return "{:.{}f}".format(n, precision)


# Печать результата в консоль или файл через writer
def print_result(result, writer):
    separator = '=' * 50  # Разделительная линия для читаемости
    line = f"\n\n{separator}\n\n".join(result)  # Соединяем результаты через разделитель
    writer.write(f'\n{separator}\n\n{line}\n\n{separator}\n')  # Выводим через выбранного writer-а


# Создание объекта для чтения данных (с консоли или из файла)
def create_reader():
    intput_mode = choose_options('Выберите способ ввода', IO_METHODS)  # Спрашиваем пользователя
    reader = ConsoleReader()  # По умолчанию — чтение с консоли
    if intput_mode == 2:
        filename = read_filename('r')  # Спрашиваем имя файла
        reader = FileReader(filename)  # Читаем из файла
    return reader


# Создание объекта для записи данных (в консоль или файл)
def create_writer():
    output_mode = choose_options('Выберите способ вывода ответа', IO_METHODS)  # Спрашиваем пользователя
    writer = ConsoleWriter()  # По умолчанию — вывод в консоль
    if output_mode == 2:
        filename = read_filename('w')  # Спрашиваем имя файла
        writer = FileWriter(filename)  # Записываем в файл
    return writer


# Функция для выбора одного из предложенных вариантов
def choose_options(message, options):
    options_str = ''.join(f'{i + 1} -> {val}\n' for i, val in enumerate(options))[:-1]  # Формируем текст вариантов
    print(f'{message}:\n{options_str}')  # Выводим варианты на экран

    result = None
    while result is None:
        try:
            result = int(input())  # Пользователь вводит номер варианта
            if result not in range(1, len(options) + 1):
                print(f'Выберите один из вариантов:\n{options_str}')
                result = None
                continue
            break
        except:
            print('Значение должно быть числом. Попробуйте снова')
    return result


# Чтение имени файла (с проверкой существования/создания)
def read_filename(mode):
    filename = None
    while filename is None:
        filename = input('Введите имя файла: ').strip()
        try:
            open(filename, mode).close()  # Пробуем открыть файл
        except:
            filename = None  # Если ошибка — просим ввести снова
            print('Не удалось найти файл!')
    return filename


# Чтение точек (x, y) от пользователя или из файла
def read_points(reader):
    x = []  # Список для хранения всех x
    y = []  # Список для хранения всех y
    while True:
        try:
            s = reader.read()  # Считываем строку

            # Если введено 'q' или пустая строка (для файлового чтения), заканчиваем ввод
            if s == 'q' or s == '' and isinstance(reader, FileReader):
                break

            xi, yi = list(map(float, s.split()))  # Преобразуем введенные значения в числа
            x.append(xi)  # Добавляем в список x
            y.append(yi)  # Добавляем в список y
        except:
            message = 'Некорректный ввод'  # Сообщение об ошибке
            if isinstance(reader, FileReader):
                raise Exception(message)  # Если ошибка в файле — прерываем выполнение
            else:
                print(message)  # Если ошибка с консоли — просим пользователя повторить ввод

    return x, y  # Возвращаем считанные точки


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


# === Переменные, описывающие доступные типы аппроксимаций ===

# Список функций аппроксимации, которые будут применяться к данным
APPROXIMATION_FUNCTIONS = [
    linear_approximation,  # Линейная аппроксимация: phi(x) = a*x + b
    square_approximation,  # Квадратичная аппроксимация: phi(x) = a2*x^2 + a1*x + a0
    cubic_approximation,  # Кубическая аппроксимация: phi(x) = a3*x^3 + a2*x^2 + a1*x + a0
    exponential_approximation,  # Экспоненциальная аппроксимация: phi(x) = b * e^(a*x)
    logarithmic_approximation,  # Логарифмическая аппроксимация: phi(x) = a * ln(x) + b
    power_approximation  # Степенная аппроксимация: phi(x) = a * x^b
]

# Список названий соответствующих функций для вывода пользователю
APPROXIMATION_FUNCTIONS_NAMES = [
    'Линейная',
    'Полиноминальная 2-й степени',
    'Полиноминальная 3-й степени',
    'Экспоненциальная',
    'Логарифмическая',
    'Степенная'
]

# Доступные способы ввода/вывода данных
IO_METHODS = ['Консоль', 'Файл']


# === Функция для построения графиков аппроксимаций ===
def draw_plot(x, y, phis, names):
    x = np.array(x)  # Преобразуем список x в массив numpy
    y = np.array(y)  # Преобразуем список y в массив numpy

    # Определяем минимальное и максимальное значение x и y
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    # Задаём небольшие отступы от краев для красоты графика
    x_margin = (x_max - x_min) * 0.1 if x_min != x_max else 1
    y_margin = (y_max - y_min) * 0.1 if y_min != y_max else 1

    plt.figure(figsize=(8, 6))  # Создаём новую фигуру для графика с размерами 8x6 дюймов

    # Рисуем точки исходных данных
    plt.scatter(x, y, color='blue', label='Точки (x, y)')

    # Создаём много промежуточных точек для плавных линий
    x_smooth = np.linspace(x_min - x_margin, x_max + x_margin, 1000)

    # Для каждой аппроксимирующей функции рисуем график
    for i, phi in enumerate(phis):
        y_smooth = phi(x_smooth)  # Вычисляем значения функции для x_smooth
        plt.plot(x_smooth, y_smooth, label=names[i])  # Строим кривую аппроксимации

    # Устанавливаем границы осей с отступами
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)

    # Подписи осей
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()  # Легенда графика
    plt.grid(True)  # Сетка для удобства восприятия
    plt.tight_layout()  # Автоматическое распределение элементов на графике
    plt.show()  # Отображение графика


def main():
    # Создание объекта для чтения данных (с консоли или из файла)
    reader = create_reader()

    # Если выбран ввод с консоли, вывести пользователю инструкцию
    if isinstance(reader, ConsoleReader):
        print('Вводите точки, по одной в строке. По окончании ввода введите q')

    # Чтение точек (x, y) из выбранного источника
    x, y = read_points(reader)
    n = len(x)  # Вычисляем количество точек

    # Списки для хранения построенных функций и их названий
    phis = []
    phis_names = []

    # Переменные для определения наилучшей аппроксимации
    max_abs_r2 = 0  # Максимальное значение коэффициента детерминации
    best_approximation_function_index = None  # Индекс лучшей аппроксимации

    result = []  # Список для сбора текстовых результатов аппроксимаций

    # Перебор всех видов аппроксимации, объявленных в списке APPROXIMATION_FUNCTIONS
    for i in range(len(APPROXIMATION_FUNCTIONS)):
        f = APPROXIMATION_FUNCTIONS[i]  # Берём функцию аппроксимации
        name = APPROXIMATION_FUNCTIONS_NAMES[i]  # И её название для отображения

        log = [f'Аппроксимирующая функция: {name}']  # Начинаем запись лога результата

        try:
            # Пытаемся построить аппроксимирующую функцию и получить её коэффициенты
            phi, c = f(x, y, n)
        except Exception as e:
            # В случае ошибки записываем её в лог и переходим к следующей функции
            log.append(f'ОШИБКА: {e}')
            result.append("\n".join(log))
            continue

        # Сохраняем успешно построенные функции и их имена для дальнейшей работы
        phis.append(phi)
        phis_names.append(name)

        # Формируем строку красивого представления аппроксимирующей функции
        phi_str = approximation_functions_coefficients_to_str[i](c)

        # Вычисляем метрики качества аппроксимации
        mse = mean_squared_error(x, y, phi, n)  # Среднеквадратичное отклонение
        r2 = coefficient_of_determination(x, y, phi, n)  # Коэффициент детерминации R²
        s = measure_of_deviation(x, y, phi)  # Мера отклонения

        # Проверяем: если текущее приближение лучше предыдущих по R², запоминаем его
        if abs(r2) > max_abs_r2:
            max_abs_r2 = abs(r2)
            best_approximation_function_index = i

        # Добавляем всю информацию о текущей функции в лог
        log.append(f'Функция: φ(x) = {phi_str}')
        log.append(f'Среднеквадратичное отклонение: σ = {round_(mse, 3)}')
        log.append(f'Коэффициент детерминации: R² = {round_(r2, 3)}')
        log.append(f'Мера отклонения: S = {round_(s, 3)}')

        # Только для линейной аппроксимации дополнительно вычисляем коэффициент корреляции Пирсона
        if f is linear_approximation:
            r = pearson_correlation_coefficient(x, y, n)
            log.append(f'Коэффициент корреляции Пирсона: r = {round_(r, 3)}')

        # Сохраняем лог текущей аппроксимации
        result.append("\n".join(log))

    # Создаём объект для записи результатов (консоль или файл)
    writer = create_writer()

    # Печатаем или записываем результат всех аппроксимаций
    print_result(result, writer)

    # Определяем и выводим наилучшую аппроксимацию
    best_approximation_function_name = None
    if best_approximation_function_index is not None:
        best_approximation_function_name = APPROXIMATION_FUNCTIONS_NAMES[best_approximation_function_index]

    print(f'Лучшая аппроксимирующая функция: {best_approximation_function_name}')

    # Если были считаны точки, строим итоговый график
    if n != 0:
        draw_plot(x, y, phis, phis_names)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
