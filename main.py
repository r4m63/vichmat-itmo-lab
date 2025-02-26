import colors as color
from culc import input_from_file as iff, input_from_console as ifc

print(color.BOLD + color.RED, 'Решение Системы Уравнений методом Гаусса!', color.END)

if __name__ == "__main__":
    while True:
        try:
            print('\n', 'Доступные функции программы:')
            print(color.GREEN,
                  '\t', '1: Считывание линейной системы из файла.')
            print('\t', '2: Ввод линейной системы.')
            print('\t', '3: Выход.', color.END)
            param = int(input('Введите число функции = '))

            if param == 1:
                print(color.UNDERLINE + color.YELLOW, 'Выбран способ считывание с файла.', color.END)
                print('Файл должен содержать линейную систему вида (Размерность не более n = 20):', '\n',
                      '\t', 'a11 a12 ... a1n | b1', '\n',
                      '\t', 'a21 a22 ... a2n | b2', '\n',
                      '\t', '... ... ... ... | ..', '\n',
                      '\t', 'an1 an2 ... ann | bn')
                iff(input('Введите путь к фалу: ').strip())
            elif param == 2:
                print(color.UNDERLINE + color.YELLOW, 'Выбран способ ввода вручную.', color.END)
                ifc()
            elif param == 3:
                print(color.BOLD + color.PURPLE, 'Конец', color.END)
                break
            else:
                print(color.BOLD + color.RED, 'Неправильно введено значение! Попробуйте снова.', color.END)
        except KeyboardInterrupt:
            print(color.BOLD + color.RED, 'Прерывание', color.END)
        except:
            print(color.BOLD + color.RED, 'Ошибка', color.END)
