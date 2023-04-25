import os
import numpy as np
import matplotlib.pyplot as plt

for i in range(0,3):    # Specify iterations
    max_devi_f_values = []
    for j in range(20):    # 20 tasks in iter*/01.model_devi
        directory = "./iter.{:06d}/01.model_devi/task.000.{:06d}".format(i, j%20)
        file_path = os.path.join(directory, "model_devi.out")
        data = np.genfromtxt(file_path, skip_header=1, usecols=4)
        max_devi_f_values.append(data)

    # Convert the list to a numpy array
    max_devi_f_values = np.concatenate(max_devi_f_values)

    # Use numpy.histogram() to calculate the frequency of each value
    hist, bin_edges = np.histogram(max_devi_f_values, range=(0, 0.2), bins=100)

    # Normalize the frequency to percentage
    hist = hist / len(max_devi_f_values) * 100

    # Use matplotlib.pyplot.plot() to plot the frequency of each value
    plt.plot(bin_edges[:-1], hist,label = 'iter{:02d}'.format(i))
    plt.legend()
    plt.xlabel("Max_devi_f eV/Å")
    plt.ylabel("Distribution %")

    with open(f'./iter{i:02d}_dist-max-devi-f.txt'.format(i), 'w') as f:
        f.write("{:>12} {:>12}\n".format("bin_edges", "hist"))
        for x, y in zip(bin_edges[:-1], hist):
            f.write('{:>12.3f} {:>12.3f}\n'.format(x, y))

plt.savefig('max-devi-f.png',dpi=300)