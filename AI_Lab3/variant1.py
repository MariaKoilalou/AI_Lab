import numpy as np
import matplotlib.pyplot as plt
import time


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


# suggest a range of values for the user input
x_range = (-5, 5)
y_range = (-5, 5)
mutation_strength_range = (0.01, 0.5)
mutation_probability_range = (0.01, 0.5)
num_generations_range = (100, 1000)

# print the suggested ranges to the user
print("Suggested parameter ranges:")
print(f"x range: {x_range}")
print(f"y range: {y_range}")
print(f"mutation strength range: {mutation_strength_range}")
print(f"mutation probability range: {mutation_probability_range}")
print(f"number of generations range: {num_generations_range}")

# Get user input for the parameters
x_min = float(input("Enter the minimum value of x: "))
x_max = float(input("Enter the maximum value of x: "))
y_min = float(input("Enter the minimum value of y: "))
y_max = float(input("Enter the maximum value of y: "))
sigma = float(input("Enter the mutation strength (sigma): "))
mut_prob = float(input("Enter the mutation probability: "))
n_generations = int(input("Enter the number of generations: "))

# Initialize the population
x_init = np.random.uniform(x_min, x_max)
y_init = np.random.uniform(y_min, y_max)

# Measure the optimization time
start_time = time.time()

# Run the algorithm
best_fitness, best_x, best_y = evolve(rosenbrock, x_init, y_init, sigma, mut_prob, n_generations)

# Calculate the optimization time
end_time = time.time()
opt_time = end_time - start_time

# Print the results
print(f"Best fitness: {best_fitness}")
print(f"Best x: {best_x}")
print(f"Best y: {best_y}")
print(f"Optimization time: {opt_time:.6f} seconds")

# Plot the optimization process
x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)
plt.contour(X, Y, Z, levels=50, cmap='cool')
plt.plot([best_x], [best_y], 'r*', markersize=10)
plt.title("Evolutionary Strategy (1+1) for Rosenbrock Function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
