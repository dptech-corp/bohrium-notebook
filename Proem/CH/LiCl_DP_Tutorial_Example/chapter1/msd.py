import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('licl.msd', skiprows=2)

time = data[:, 0]
msd1 = data[:, 1]
msd2 = data[:, 2]

plt.plot(time/1000, msd1, 'b-', label='Li+') # 1fs= 1/1000ps
plt.plot(time/1000, msd2, 'r-', label='Cl-')
plt.xlabel('time(ps)') 
plt.ylabel('MSD(Å^2)')

slope1, residuals = np.polyfit(time, msd1, 1)
slope2, residuals = np.polyfit(time, msd2, 1)

Diff1 = slope1/6 * 1e-5  # D=1/6*slope; 1 A^2/fs= 1e-5 m^2/s
Diff2 = slope2/6 * 1e-5

print(f"Diffusion Coefficients of Li+: {Diff1} m^2/s")
print(f"Diffusion Coefficients of Cl-: {Diff2} m^2/s")

plt.legend()
plt.savefig('msd.png',dpi=300)

