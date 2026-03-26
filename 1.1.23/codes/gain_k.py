import numpy as np
import control as ct
import matplotlib.pyplot as plt

# G(s) = K / [s(s+3)(s+4)]
num = [1]
den = [1, 7, 12, 0] # s^3 + 7s^2 + 12s
sys = ct.TransferFunction(num, den)

# Find poles
s_target = -1 + 1j
poles = ct.poles(sys)

# K = Product of distances from each pole to the target point
distances = [np.abs(s_target - p) for p in poles]
K_calc = np.prod(distances)

# Visualization
plt.figure(figsize=(9, 6))

# Root Locus
roots, gains = ct.root_locus(sys, plot=False)

# Plot Root Locus Path
for i in range(roots.shape[1]):
    lbl = 'Root Locus Path' if i == 0 else ""
    plt.plot(np.real(roots[:, i]), np.imag(roots[:, i]), color='C0', lw=1.5, label=lbl, zorder=2)

# Plot Open-Loop Poles
plt.plot(np.real(poles), np.imag(poles), 'x', color='C0', markersize=10, mew=2, label='Poles', zorder=3)

# Plot Target Point
plt.plot(s_target.real, s_target.imag, 'ro', markersize=8, label=f'Target Point', zorder=4)

for p in poles:
    x_coord = np.real(p)
    y_coord = np.imag(p)
    # Using :g formats it cleanly without trailing zeros
    plt.text(x_coord, y_coord + 0.2, f'({x_coord:g}, {y_coord:g})',
             color='C0', fontsize=10, fontweight='bold', ha='center')

t_x = s_target.real
t_y = s_target.imag
plt.text(t_x + 0.15, t_y + 0.15, f'({t_x:g}, {t_y:g})',
         color='red', fontsize=10, fontweight='bold', va='bottom', ha='left')

plt.title("Root Locus: Determination of Gain K")
plt.xlabel("Real Axis ($\sigma$)")
plt.ylabel("Imaginary Axis ($j\omega$)")

plt.axhline(0, color='black', lw=0.5, alpha=0.3, zorder=1)
plt.axvline(0, color='black', lw=0.5, alpha=0.3, zorder=1)
plt.grid(True, linestyle=':', alpha=0.5)

plt.xlim([-6, 2])
plt.ylim([-4, 4])
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

print(f"Target Point Coordinates: ({s_target.real:g}, {s_target.imag:g})")
print(f"Pole Coordinates: {[(np.real(p), np.imag(p)) for p in poles]}")
print(f"Calculated Gain K: {K_calc:.2f}")
