from matplotlib import pyplot as plt
import numpy as np
import csv

# Lists hold the mean and standard deviations for each time step for the five functions
value = []
quarter = []
closest = []
sqrt = []
normal = []

# List of lists, labels to read the file, labels for the plot, and colors for all five methods
methods_long = [closest, normal, sqrt, quarter, value]
labels_in_long = ['Closest', 'Normal', 'Square_root', 'Quarter', 'Value']
labels_out_long = ['Closest First', 'Value / Distance', 'Value / sqrt(Distance)', 'Value / (Distance)^1/4', 'Value']
colors_long = ['darkorange', 'deepskyblue', 'b', 'g', 'mediumorchid']

# List of lists, labels to read the file, labels for the plot, and colors for only the closest and normal methods
methods_short = [closest, normal]
labels_in_short = ['Closest', 'Normal']
labels_out_short = ['Closest First', 'Value / Distance']
colors_short = ['darkorange', 'deepskyblue']

with open('formatted_fast_toploss_score24recursive.csv') as file:

    readCSV = csv.reader(file, delimiter=',')
    for row in readCSV:
        for i in range(len(methods_long)):
            if row[0] == labels_in_long[i]:
                methods_long[i].append(row[2:])


def make_plot(methods, labels_out, colors, name=None):
    # Counter provides x-labels
    counter = range(499)
    
    # For each method
    for k in range(len(methods)):

        # Convert average and stdevs to lists of floats
        y = list(map(float, methods[k][0]))
        dev = list(map(float, methods[k][1]))

        # Initialize arrays for one standard deviation in each direction
        y_low = []
        y_high = []

        # Calculate the low and high at each time step, setting to 1 or 0 for max / min
        for j in range(len(y)):
            y_min = y[j] - dev[j]
            if y_min < 0:
                y_min = 0

            y_low.append(y_min)
            y_max = y[j] + dev[j]

            if y_max > 1:
                y_max = 1

            y_high.append(y_max)

        # Add to plot
        plt.plot(counter, y, label=labels_out[k], color=colors[k])
        plt.fill_between(counter, y_low, y_high, color=colors[k], alpha=.1)

    plt.legend(loc="lower right")
    plt.title("Average and Standard Deviation for Percent of Explored Area")
    plt.xlabel("Time Steps")
    plt.ylabel("Percent of Explored Area")

    # If no file name was passed, show the plot. Otherwise, save as that file name
    if not name:
        plt.show()
    else:
        plt.savefig(name)


if __name__ == "__main__":
    make_plot(methods_short, labels_out_short, colors_short, 'Percent_found_two_variable1_24.png')
    #make_plot(methods_long, labels_out_long, colors_long, 'Percent_found_all.png')
