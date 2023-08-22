import matplotlib.pyplot as plt


def units_plot(coordinates, unit_letter_dict=None, units=None):
    x = [i[0] for i in coordinates]
    y = [i[1] for i in coordinates]
    if unit_letter_dict is None:
        point_names = [i for i in range(len(coordinates))]
    if units is None:
        point_names = [f'{i}{unit_letter_dict[i]}' for i in sorted(unit_letter_dict.keys())]

    for i, name in enumerate(point_names):
        plt.annotate(name, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.scatter(x, y, label='Units', color='blue', marker='o')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Units on a Coordinate Plane')
    plt.legend()
    plt.grid(True)
    plt.show()
