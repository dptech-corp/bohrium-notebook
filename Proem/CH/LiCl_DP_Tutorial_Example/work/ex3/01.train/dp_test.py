import numpy as np
import matplotlib.pyplot as plt

def plot_scatter(ax, data, x_key, y_key, xlabel, ylabel):
    ax.scatter(data[x_key], data[y_key], label=y_key)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

data_e = np.genfromtxt("results.e.out", names=["data_e", "pred_e"])
data_f = np.genfromtxt("results.f.out", names=["data_fx", "data_fy", "data_fz", "pred_fx", "pred_fy", "pred_fz"])

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

plot_scatter(axs[0], data_e, 'data_e', 'pred_e', 'DFT energy (eV)', 'DP energy (eV)')

for force_direction in ['fx', 'fy', 'fz']:
    plot_scatter(axs[1], data_f, f'data_{force_direction}', f'pred_{force_direction}', 'DFT force (eV/Å)', 'DP force (eV/Å)')

plt.tight_layout()
plt.savefig('DP&DFT_combined_horizontal.png', dpi=300)

