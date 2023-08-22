import numpy as np
from scipy.spatial import distance_matrix
from plot import units_plot


class FrequencyDistributor:
    def __init__(
            self, unit_list: list, frequency_list: list,
            coordinates: list):
        # Список БМ
        self.unit_list = unit_list

        # Количество БМ
        self.unit_count = len(unit_list)

        # Список частот
        self.frequency_list = frequency_list

        # Словарь, где ключ - БМ, а значение - присвоенная литера частоты
        self.unit_letter_dict = {}

        # Координаты БМ
        self.coordinates = np.array(coordinates)

        # Матрица расстояний между БМ
        self.distance_matrix = (
            distance_matrix(self.coordinates, self.coordinates)
            .astype(np.uint32)
        )

        # Распределение частот по БМ
        self.distribute()

    def distribute(self):
        # Если количество БМ меньше 4, то по порядку
        # присваиваем каждой БМ свою частоту
        if self.unit_count < 4:
            for index, value in enumerate(self.unit_list):
                self.unit_letter_dict[value] = self.frequency_list[index]
            return

        # Создание матрицы с маской для исключения нулевых значений
        # (т.к. нулевые значения - это расстояние от БМ до самой себя)
        masked_matrix = np.ma.masked_equal(self.distance_matrix, 0)

        # Поиск минимального расстояния в матрице
        min_distance = masked_matrix.min()

        # Поиск индексов минимального расстояния в матрице;
        # индексы соответствуют индексам БМ в списке unit_list
        min_distance_indexes = np.where(masked_matrix == min_distance)[0]

        # Запись в словарь БМ с минимальным расстоянием друг от друга
        self.unit_letter_dict[self.unit_list[min_distance_indexes[0]]] = (
            frequency_list[0])
        self.unit_letter_dict[self.unit_list[min_distance_indexes[1]]] = (
            frequency_list[1])

        # Маскирование минимального расстояния в матрице
        # (т.к. оно помешает найти минимальное расстояние до третьей БМ)
        masked_matrix.mask |= (masked_matrix == min_distance)

        # Минимальное расстояние до БМ, которая находится рядом с двумя
        # БМ с минимальным расстоянием друг от друга
        third_unit_min_distance = masked_matrix[min_distance_indexes, :].min()

        # Индекс БМ, которая находится рядом с двумя
        third_unit_index = np.setdiff1d(
            np.where(masked_matrix == third_unit_min_distance)[0],
            min_distance_indexes
        )[0]

        # Запись в словарь БМ, которая находится рядом с двумя
        self.unit_letter_dict[self.unit_list[third_unit_index]] = (
            frequency_list[2])

        # Индексы первой тройки БМ
        first_three_units_indexes = np.concatenate(
            (min_distance_indexes, [third_unit_index]))

        # Максимальное расстояние до первой тройки БМ
        fourth_unit_max_distance = (
            masked_matrix[first_three_units_indexes, :].max())

        # Индекс БМ, которая находится дальше всех от первой тройки БМ
        fourth_unit_index = np.setdiff1d(
            np.where(masked_matrix == fourth_unit_max_distance)[0],
            first_three_units_indexes
        )[0]

        # Индекс БМ, которая находится дальше всего от четвертой БМ
        unit_index_paired_with_fourth_unit = np.intersect1d(
            np.where(masked_matrix == fourth_unit_max_distance)[0],
            first_three_units_indexes
        )[0]

        # Запись в словарь БМ, которая находится дальше всех от первой тройки
        self.unit_letter_dict[self.unit_list[fourth_unit_index]] = (
            self.unit_letter_dict[self.unit_list[unit_index_paired_with_fourth_unit]])

        if self.unit_count > 4:
            # Маскирование четвертой БМ в матрице расстояний, чтобы исключить
            # ее из поиска максимального расстояния до пятой БМ
            fourth_unit_index_column_mask = np.zeros(
                masked_matrix.shape, dtype=bool)
            fourth_unit_index_column_mask[:, fourth_unit_index] = True
            masked_matrix.mask |= fourth_unit_index_column_mask

            # Максимальное расстояние до пятой БМ (поиск по первой тройке БМ,
            # исключая находящуюся в группе с четвертой БМ)
            fifth_unit_max_distance = (
                masked_matrix[first_three_units_indexes[
                                  first_three_units_indexes
                                  != unit_index_paired_with_fourth_unit], :]
                .max()
            )

            # Индекс пятой БМ
            fifth_unit_index = np.setdiff1d(
                np.where(masked_matrix == fifth_unit_max_distance)[0],
                first_three_units_indexes
            )[0]

            # Индекс БМ, у которой максимальное расстояние до пятой БМ
            unit_index_paired_with_fifth_unit = np.intersect1d(
                np.where(masked_matrix == fifth_unit_max_distance)[0],
                first_three_units_indexes
            )[0]

            # Присвоение пятой БМ литеры частоты
            self.unit_letter_dict[self.unit_list[fifth_unit_index]] = (
                self.unit_letter_dict[self.unit_list[unit_index_paired_with_fifth_unit]]
            )

        if self.unit_count > 5:
            # Возвращение четвертой БМ в матрицу расстояний
            masked_matrix[:, fourth_unit_index] = (
                masked_matrix[:, fourth_unit_index].data)

            # Индексы всех БМ
            a = np.arange(masked_matrix.shape[0])

            # Индексы первой тройки БМ, четвертой и пятой БМ
            b = np.concatenate((
                first_three_units_indexes,
                [fourth_unit_index],
                [fifth_unit_index]
            ))

            # Индекс шестой БМ (оставшейся)
            sixth_unit_index = np.setdiff1d(a, b)[0]

            # Индекс БМ из первой тройки, у которой нет пары
            remaining_unit_index = np.setdiff1d(
                first_three_units_indexes,
                [
                    unit_index_paired_with_fourth_unit,
                    unit_index_paired_with_fifth_unit
                ]
            )[0]

            # Расстояние от БМ без пары до шестой БМ
            dist_to_remaining_unit = (
                masked_matrix[remaining_unit_index][sixth_unit_index])

            # Расстояние от шестой БМ до ближайшей к ней БМ
            distance_to_near_unit = masked_matrix[sixth_unit_index].min()

            # Индекс ближайшей к шестой БМ БМ
            near_unit_index = np.where(
                masked_matrix[sixth_unit_index] == distance_to_near_unit)[0][0]

            # Расстояние от ближайшей к шестой БМ БМ до самой дальней от нее БМ
            nearest_unit_distance_to_furthest_unit = (
                masked_matrix[near_unit_index].max())

            # Индекс самой дальней от ближайшей к шестой БМ БМ
            furthest_unit_index = np.where(
                masked_matrix[near_unit_index]
                == nearest_unit_distance_to_furthest_unit
            )[0][0]

            # Присвоение шестой БМ литеры частоты: если расстояние от БМ без пары
            # до шестой БМ больше расстояния от ближайшей к шестой БМ БМ до самой
            # дальней от нее БМ, то шестой БМ получает литеру частоты БМ без пары,
            # иначе - литеру частоты самой дальней от ближайшей к шестой БМ БМ
            if dist_to_remaining_unit > nearest_unit_distance_to_furthest_unit:
                self.unit_letter_dict[self.unit_list[sixth_unit_index]] = (
                    self.unit_letter_dict[self.unit_list[remaining_unit_index]])
            else:
                self.unit_letter_dict[self.unit_list[sixth_unit_index]] = (
                    self.unit_letter_dict[self.unit_list[furthest_unit_index]])


if __name__ == "__main__":
    frequency_list = ['a', 'b', 'c']

    units1 = [0, 1, 2, 3, 4, 5]
    coordinates1 = [
        [162, 522],
        [1, 812],
        [67, 432],
        [252, 361],
        [791, 71],
        [512, 213]
    ]
    units_plot(coordinates1, units=units1)
    distributor = FrequencyDistributor(units1, frequency_list, coordinates1)
    units_plot(coordinates1, unit_letter_dict=distributor.unit_letter_dict)

    units2 = [0, 1, 2]
    coordinates2 = [
        [29, 28],
        [899, 884],
        [880, 297]
    ]
    units_plot(coordinates2, units=units2)
    distributor = FrequencyDistributor(units2, frequency_list, coordinates2)
    units_plot(coordinates2, unit_letter_dict=distributor.unit_letter_dict)

    units3 = [0, 1, 2, 3]
    coordinates3 = [
        [571, 902],
        [617, 154],
        [201, 470],
        [380, 321]
    ]
    units_plot(coordinates3, units=units3)
    distributor = FrequencyDistributor(units3, frequency_list, coordinates3)
    units_plot(coordinates3, unit_letter_dict=distributor.unit_letter_dict)

    units4 = [0, 1, 2, 3, 4]
    coordinates4 = [
        [108, 144],
        [309, 855],
        [806, 288],
        [656, 968],
        [973, 674]
    ]
    units_plot(coordinates4, units=units4)
    distributor = FrequencyDistributor(units4, frequency_list, coordinates4)
    units_plot(coordinates4, unit_letter_dict=distributor.unit_letter_dict)
