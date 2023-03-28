import numpy as np
import matplotlib.pyplot as plt


# Define the Rosenbrock function
def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


# Define the Evolutionary Strategy (1+1) algorithm
def evolve(fitness, x_init, y_init, sigma, mut_prob, n_generations):
    x = x_init
    y = y_init
    best_fitness = fitness(x, y)
    best_x = x
    best_y = y
    for i in range(n_generations):
        # Mutate
        x_new = x + sigma * np.random.randn()
        y_new = y + sigma * np.random.randn()
        # Evaluate fitness of mutated offspring
        fitness_new = fitness(x_new, y_new)
        # Compare fitness of offspring with parent
        if fitness_new < best_fitness:
            best_fitness = fitness_new
            best_x = x_new
            best_y = y_new
            x = x_new
            y = y_new
        else:
            x = x
            y = y
        # Update sigma and mutation probability
        sigma *= np.exp(-0.2 * i / n_generations)
        mut_prob *= np.exp(0.05 * np.random.randn())
    return best_fitness, best_x, best_y


# Define the range of values from which the population is initialized
x_range = (-5, 5)
y_range = (-5, 5)

# Define the mutation strength and mutation probability
sigma = 0.1
mut_prob = 0.5

# Define the number of generations
n_generations = 100

# Initialize the population
x_init = np.random.uniform(*x_range)
y_init = np.random.uniform(*y_range)

# Run the algorithm
best_fitness, best_x, best_y = evolve(rosenbrock, x_init, y_init, sigma, mut_prob, n_generations)

# Print the results
print(f"Best fitness: {best_fitness}")
print(f"Best x: {best_x}")
print(f"Best y: {best_y}")

# Plot the optimization process
x = np.linspace(*x_range, 100)
y = np.linspace(*y_range, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)
plt.contour(X, Y, Z, levels=50, cmap='cool')
plt.plot([best_x], [best_y], 'r*', markersize=10)
plt.title("Evolutionary Strategy (1+1) for Rosenbrock Function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
