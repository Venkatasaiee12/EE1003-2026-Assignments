import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define the variables
x_var = cp.Variable(3)

# Define Matrix V and vector n
V_mat = np.diag([2, 3, 4])
n_vec = np.array([2, 3, 4])

# Maximize n^T x subject to x^T V x <= 1
objective = cp.Maximize(n_vec @ x_var)
constraints = [cp.quad_form(x_var, V_mat) <= 1]
prob = cp.Problem(objective, constraints)

# Solve the problem
prob.solve()

# Results
c_max = prob.value
q_max = x_var.value
q_min = -q_max  # Symmetry of the ellipsoid

print(f"Max value found: {c_max:.4f}")
print(f"Point of maximization: {q_max}")

# 2. Visualization setup
fig = plt.figure(figsize=(5, 3))
ax = fig.add_subplot(111, projection='3d')

# Generate Ellipsoid Data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
rx, ry, rz = 1/np.sqrt(2), 1/np.sqrt(3), 1/np.sqrt(4)
x_ell = rx * np.outer(np.cos(u), np.sin(v))
y_ell = ry * np.outer(np.sin(u), np.sin(v))
z_ell = rz * np.outer(np.ones(np.size(u)), np.cos(v))

# Generate Tangent Planes
grid = np.linspace(-1.2, 1.2, 10)
X, Y = np.meshgrid(grid, grid)
Z_max = (c_max - n_vec[0]*X - n_vec[1]*Y) / n_vec[2]
Z_min = (-c_max - n_vec[0]*X - n_vec[1]*Y) / n_vec[2]

# Plot Ellipsoid
ax.plot_surface(x_ell, y_ell, z_ell, color='cyan', alpha=0.2, linewidth=0)

# Plot Planes
ax.plot_surface(X, Y, Z_max, color='orange', alpha=0.4)
ax.plot_surface(X, Y, Z_min, color='blue', alpha=0.4)

# Mark the points with reduced size
ax.scatter(q_max[0], q_max[1], q_max[2], color='red', s=40, edgecolors='k')
ax.scatter(q_min[0], q_min[1], q_min[2], color='blue', s=40, edgecolors='k')

# Label coordinates at the dots
ax.text(q_max[0], q_max[1], q_max[2]+0.1,
        f"({q_max[0]:.2f}, {q_max[1]:.2f}, {q_max[2]:.2f})",
        color='red', fontsize=9, fontweight='bold', ha='center')
ax.text(q_min[0], q_min[1], q_min[2]-0.2,
        f"({q_min[0]:.2f}, {q_min[1]:.2f}, {q_min[2]:.2f})",
        color='blue', fontsize=9, fontweight='bold', ha='center')

# Set view to between the planes
ax.view_init(elev=5, azim=150)

# Legend
legend_elements = [
    Line2D([0], [0], color='cyan', lw=4, alpha=0.3, label=r'$2x^2 + 3y^2 + 4z^2 = 1$'),
    Line2D([0], [0], color='orange', lw=4, label=r'$2x + 3y + 4z = 3$'),
    Line2D([0], [0], color='blue', lw=4, label=r'$2x + 3y + 4z = -3$')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Planes touching the ellipsoid')

plt.show()
