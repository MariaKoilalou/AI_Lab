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
        # Mutate with mutation step size
        x_new = x + sigma * np.random.randn()
        y_new = y + sigma * np.random.randn()
        # Clip to input ranges
        x_new = np.clip(x_new, x_min, x_max)
        y_new = np.clip(y_new, y_min, y_max)
        sigma = np.clip(sigma, mutation_strength_range[0], mutation_strength_range[1])
        mut_prob = np.clip(mut_prob, mutation_probability_range[0], mutation_probability_range[1])
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
        # The factor 0.2 controls the rate at which sigma decreases
        sigma *= np.exp(-0.2 * i / n_generations)
        # The factor 0.05 controls the magnitude of this noise.
        mut_prob *= np.exp(0.05 * np.random.randn())
    return best_fitness, best_x, best_y


# Suggest a range of values for the user input
x_range = (-5, 5)
y_range = (-5, 5)
mutation_strength_range = (0.01, 0.5)
mutation_probability_range = (0.01, 0.5)
num_generations_range = (100, 1000)

# Print the suggested ranges to the user
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

# Plot the optimization process in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([best_x], [best_y], [best_fitness], 'r*', markersize=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('fitness')
ax.set_title('Evolutionary Strategy (1+1) for Rosenbrock Function')

# Create meshgrid for 3D plot
x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='cool', alpha=0.8)

plt.show()
