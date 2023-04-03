import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2


def optimize():
    # Initialize random starting coordinates
    x = random.uniform(-5, 5)
    y = random.uniform(-5, 5)

    # Set initial step size
    sigma = 0.1

    # Evaluate initial solution
    f = rosenbrock(x, y)

    # Save history for visualization
    x_hist = [x]
    y_hist = [y]
    f_hist = [f]

    # Repeat until convergence
    while sigma > 1e-6:
        # Apply mutation with Gaussian noise
        new_x = x + sigma * random.gauss(0, 1)
        new_y = y + sigma * random.gauss(0, 1)

        # Evaluate new solution
        new_f = rosenbrock(new_x, new_y)

        # If new solution is better, replace old solution
        if new_f < f:
            x, y = new_x, new_y
            f = new_f

        # Reduce step size
        sigma *= 0.95

        # Save history for visualization
        x_hist.append(x)
        y_hist.append(y)
        f_hist.append(f)

    # Return optimal solution and function value
    return x, y, f, x_hist, y_hist, f_hist


best_x, best_y, f, x_hist, y_hist, f_hist = optimize()

print(f"Optimal solution found: x={best_x}, y={best_y}, value={f}")

# Plot the Rosenbrock function
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = y = np.arange(-5.0, 5.0, 0.1)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

# Plot the search history
ax.plot(x_hist, y_hist, f_hist, color='red', linewidth=2)

plt.show()
