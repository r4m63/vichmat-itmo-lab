import colors as color
import numpy as np

def input_from_file(path):
    try:
        n = 0
        a = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                if (line != '\n') and (line != ' ') and (line != ' \n'):
                    n += 1
        if n > 20:
            print(color.BOLD + color.RED, 'В файле больше 20 уравнений! Уменьшите и попробуйте снова.', color.END)
            return
        file.close()
        with open(path, 'r', encoding='utf-8') as file:
            for row in file:
                line = list(row.split())
                if (line[-2] != '|') or (len(line) - 2 != n):
                    print(color.BOLD + color.RED, 'Файл имеет ошибку формата! Исправьте и попробуйте снова.', color.END)
                    return
                a.append(list(line))
        file.close()
        calculator = Calculator(n, optimize(a, n))
        calculator.calculate()
        del calculator
    except FileNotFoundError:
        print(color.BOLD + color.RED, 'Файл не был найден по пути: ', color.END, path)


def input_from_console():
    try:
        n = int(input('Введите кол-во уравнений не более 20: '))
        if (n <= 20) or (n > 1):
            a = []
            print('Введите коэффициенты уравнения в формает:')
            print(color.YELLOW, '\t', 'ai1 ai2 ... aij | bi', color.END)
            for i in range(n):
                while True:
                    line = list((input(str(i + 1) + ': ').split()))
                    if (int(len(line)) - 2 != n) or (line[-2] != '|'):
                        print(color.BOLD + color.RED, 'Кол-во строк не равно кол-ву столбцов. Или неправильный формат.',
                              color.END)
                        print('Попробуйте снова')
                    else:
                        a.append(line)
                        break
            calculator = Calculator(n, optimize(a, n))
            calculator.calculate()
            del calculator
        else:
            print(color.BOLD + color.RED, 'Неправильный ввод!', color.END)
            return
    except ValueError:
        print(color.BOLD + color.RED, 'Неправильный ввод! ', color.END)


# Делает матрицу дробными числами из строк
def optimize(arr, n):
    i = 0
    while i < n:
        j = 0
        while j < n:
            arr[i][j] = float(arr[i][j])
            j += 1
        arr[i][j + 1] = float(arr[i][j + 1])
        i += 1
    return arr


class Calculator:
    n = 0           # Количество уравнений и неизвестных
    x = []          # Вектор неизвестных
    system = []     # Система уравнений
    det = 0         # Определитель
    swap = 0        # Кол-во перестановок

    def __init__(self, n, system):
        self.n = n
        self.system = system
        self.x = []
        self.swap = 0

    def calculate(self):
        try:
            print('\n', color.UNDERLINE + color.YELLOW, 'Наша система:', color.END)
            self.__print_system()

            self.__make_triangle() # Приведение системы к треугольному виду
            print('\n', color.UNDERLINE + color.YELLOW, 'Треугольная матрица:', color.END)
            self.__print_system()

            self.__get_determinate() # Вычисление определителя

            # Обратный ход метода Гаусса
            self.__calc_vector_x()
            self.__print_vector_x()

            self.__print_vector_residuals() # Вычисление и вывод невязок

            self.__solve_with_numpy() # Решение через numpy
        except ZeroDivisionError:
            return
        except ArithmeticError:
            return

    # Проверка диагональных элементов и перестановка строк при необходимости
    def __check_diagonal(self, i):
        j = i
        while j < self.n:
            if (self.system[j][i] != 0) and (self.system[i][j] != 0):
                swap = self.system[j]
                self.system[j] = self.system[i]
                self.system[i] = swap # Перестановка строк
                self.swap += 1
                return
            j += 1
        print(color.BOLD + color.RED, 'Нет решений!', color.END)
        return ArithmeticError

    # Вычисление треугольной матрицы прямым способом по формулам
    def __make_triangle(self):
        # Прямой ход метода Гаусса: преобразование системы в треугольный вид
        try:
            i = 0
            while i < self.n:
                if self.system[i][i] == 0:
                    self.__check_diagonal(i) # Проверка и перестановка строк при нулевом диагональном элементе
                m = i
                while m < self.n - 1:
                    a = -(self.system[m + 1][i] / self.system[i][i])
                    j = i
                    while j < self.n:
                        self.system[m + 1][j] += a * self.system[i][j]
                        j += 1
                    self.system[m + 1][-1] += a * self.system[i][-1]
                    m += 1
                k = 0
                line_sum = 0
                while k < self.n:
                    line_sum += self.system[i][k]
                    k += 1
                if line_sum == 0:
                    print(color.BOLD + color.RED, 'Данная система не совместима, решений нет!', color.END)
                    return ArithmeticError
                i += 1
        except ValueError:
            print(color.BOLD + color.RED, 'Некорректная работа с данными!', color.END)
            return

    # Вывод системы на экран
    def __print_system(self):
        # print(color.UNDERLINE + color.YELLOW, 'Система уравнений:', color.END)

        for row in self.system:
            formatted_row = "  ".join(f"{float(elem):8.3f}" if elem != '|' else "|" for elem in row)
            print(formatted_row)

    # Подсчет и вывод определителя на экран
    def __get_determinate(self):
        i = 0
        self.det = 1
        while i < self.n:
            self.det *= self.system[i][i]
            i += 1
        if self.swap % 2 == 1:
            self.det *= -1
        print('\n', color.UNDERLINE + color.YELLOW, 'Определитель', color.END, ' = ', self.det, '\n')
        if self.det == 0:
            print(color.BOLD + color.RED, 'Система является вырожденной, нет решения.', color.END)
            return ArithmeticError

    # Подсчет неизвестных x1, x2, x3, ..., xn
    def __calc_vector_x(self):
        i = self.n - 2
        self.x.append(self.system[self.n - 1][-1] / self.system[self.n - 1][self.n - 1])
        while i > -1:
            k = self.n - 1
            val = self.system[i][-1]
            while k > i:
                val -= self.x[self.n - 1 - k] * self.system[i][k]
                k -= 1
            self.x.append(val / self.system[i][i])
            i -= 1

    def __print_vector_x(self):
        i = 0
        print(color.UNDERLINE + color.YELLOW, 'Решение системы:', color.END)
        self.x.reverse()
        while i < self.n:
            print('\t', 'x[' + str(i) + ']:', self.x[i])
            i += 1
        print('')

    # Подсчет коэф. невязки r1, r2, r3, ..., rn и вывод на экран
    def __print_vector_residuals(self):
        i = 0
        print(color.UNDERLINE + color.YELLOW, 'Невязки (величина ошибки):', color.END)
        while i < self.n:
            res = 0
            j = 0
            while j < self.n:
                res += self.system[i][j] * self.x[j]
                j += 1
            res -= self.system[i][-1]
            i += 1
            print('\t', 'Невязка для', i, 'строки:', abs(res))
        print('')

    def __solve_with_numpy(self):
        try:
            print(color.UNDERLINE + color.GREEN, 'Попытка решения через numpy...', color.END)

            # Преобразуем `self.system` в числовой формат, независимо от `optimize()`
            A = np.zeros((self.n, self.n))  # Матрица коэффициентов (n x n)
            B = np.zeros(self.n)  # Вектор свободных членов (n)

            for i in range(self.n):
                for j in range(self.n):
                    A[i][j] = float(self.system[i][j])  # Преобразуем коэффициенты в float
                B[i] = float(self.system[i][-1])  # Свободный член тоже в float

            # Красивый вывод матрицы A и вектора B
            self.__print_matrix(A, "Матрица A:")
            self.__print_vector(B, "Вектор B:")

            # Решаем систему уравнений
            x_numpy = np.linalg.solve(A, B)
            det_numpy = np.linalg.det(A)

            print('\n', color.UNDERLINE + color.GREEN, 'Решение через numpy:', color.END)
            for i, val in enumerate(x_numpy):
                print(f'\t x[{i}]: {val}')
            print(f'\nОпределитель через numpy: {det_numpy}')

            self.__compare_results(x_numpy, det_numpy)

        except np.linalg.LinAlgError:
            print(color.BOLD + color.RED, 'Система вырождена или несовместна.', color.END)
        except Exception as e:
            print(color.BOLD + color.RED, 'Ошибка:', str(e), color.END)

    def __compare_results(self, x_numpy, det_numpy):
        print('\n', color.UNDERLINE + color.CYAN, 'Сравнение результатов:', color.END)

        # Сравнение определителей
        print(f'Определитель (Гаусс): {self.det}, Определитель (NumPy): {det_numpy}')

        # Сравнение решений
        print('Сравнение решений:')
        for i in range(self.n):
            diff = abs(self.x[i] - x_numpy[i])
            print(f'\t Разница для x[{i}]: {diff}')

    def __print_matrix(self, matrix, title="Матрица:"):
        print(color.UNDERLINE + color.YELLOW, title, color.END)

        for row in matrix:
            formatted_row = "  ".join(f"{float(elem):8.3f}" for elem in row)
            print(formatted_row)

    def __print_vector(self, vector, title="Вектор:"):
        print(color.UNDERLINE + color.YELLOW, title, color.END)

        for elem in vector:
            print(f"{float(elem):8.3f}")