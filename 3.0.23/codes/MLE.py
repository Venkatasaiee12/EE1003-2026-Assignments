import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# Setup figure
fig, ax = plt.subplots(figsize=(14, 8))

# Define our observed data
data = np.array([3.5, 5.2, 7.5])

# Define the candidate thetas to test and their colors
thetas =[2.4, 2.7, 3.0, 3.3]
colors = ['#e74c3c', '#27ae60', '#2980b9', '#8e44ad']

x = np.linspace(1.2, 9.8, 5000)

# 4. Loop through, calculate validity, and plot using SciPy
for theta, color in zip(thetas, colors):

    is_valid = np.all((data >= theta) & (data <= theta + 5))

    status = "Valid MLE" if is_valid else "Invalid"
    dynamic_label = rf'$\theta = {theta}$ ({status})'

    # Calculate PDF using SciPy
    y = uniform.pdf(x, loc=theta, scale=5)

    # Plot the outline calculated by SciPy
    ax.plot(x, y, color=color, linewidth=2.2, alpha=0.9, label=dynamic_label)

    ax.fill_between(x, y, 0, color=color, alpha=0.08)

# Plot the observed data points
ax.scatter(data, [0, 0, 0], color='black', s=35, zorder=10,
           label=r'Observations ($x_i$)')

# Formatting
ax.set_title('Finding the Maximum likelihood Estimation (MLE) using PDF',
             fontsize=16, pad=25)
ax.set_xlabel(r'Sample Space ($x$)', fontsize=12)
ax.set_ylabel(r'Density $f(x|\theta)$', fontsize=12)

# Set axis limits
ax.set_ylim(-0.02, 0.30)

# Add a light dotted grid
ax.grid(True, linestyle=':', alpha=0.5)

# Add the legend
ax.legend(loc='upper right', framealpha=1.0, fontsize=10.5)

plt.tight_layout()
plt.show()
