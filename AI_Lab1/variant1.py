import queue
import numpy as np


# Load the maze from the file
def load_maze():
    with open('maze.txt', 'r') as f:
        maze = [list(line.strip()) for line in f.readlines()]
    return maze


def solve_maze(search_algo):
    # Visualize the maze and the path found by the algorithm
    import time

    # Visualize the maze and the path found by the algorithm
    def visualize(maze, path, visited=None):
        maze_copy = np.array(maze)
        if path is not None:
            for i, j in path:
                maze_copy[i][j] = '*'
        if visited is not None:
            for i, j in visited:
                maze_copy[i][j] = '?'
        print('\n'.join([''.join(row) for row in maze_copy]))
        time.sleep(0.05)  # Add a 1-second delay to see the maze changes

    # Find the start and end position in the maze
    def find_start_end(maze):
        for i in range(len(maze)):
            for j in range(len(maze[0])):
                if maze[i][j] == 'S':
                    start_pos = (i, j)
                elif maze[i][j] == 'E':
                    end_pos = (i, j)
        return start_pos, end_pos

    # Check if a position is valid (not outside the maze and not a wall)
    def is_valid_position(maze, pos):
        i, j = pos
        if i < 0 or i >= len(maze) or j < 0 or j >= len(maze[0]):
            return False
        if maze[i][j] == '#':
            return False
        return True

    # Get the neighbors of a position
    def get_neighbors(maze, pos):
        i, j = pos
        neighbors = []
        for ni, nj in [(i - 1, j), (i, j + 1), (i + 1, j), (i, j - 1)]:
            if is_valid_position(maze, (ni, nj)):
                neighbors.append((ni, nj))
        return neighbors

    # Breadth-First Search Algorithm
    def bfs(maze, start_pos, end_pos):
        q = queue.Queue()
        q.put(start_pos)
        visited = set()
        parent = {}
        step = 0
        while not q.empty():
            step += 1
            print(f"Step {step}:")
            new_visited = set()
            while not q.empty():
                pos = q.get()
                visited.add(pos)
                visualize(maze, None, visited)  # Visualize the maze with visited nodes
                print(f"  Visited: {pos}")
                if pos == end_pos:
                    # Found the end position, reconstruct the path
                    path = [end_pos]
                    while path[-1] != start_pos:
                        path.append(parent[path[-1]])
                    path.reverse()
                    visualize(maze, path)  # Visualize the maze with the path found
                    return path
                for neighbor in get_neighbors(maze, pos):
                    if neighbor not in visited and neighbor not in new_visited:
                        q.put(neighbor)
                        parent[neighbor] = pos
                        new_visited.add(neighbor)
                        print(f"  Added to queue: {neighbor}")
            visited |= new_visited
            visualize(maze, None, visited)  # Visualize the maze with visited nodes
        # End position is unreachable
        return None

    # Depth-First Search Algorithm
    def dfs(maze, start_pos, end_pos):
        stack = [start_pos]
        visited = set()
        parent = {}
        step = 0
        while stack:
            step += 1
            print(f"Step {step}:")
            pos = stack.pop()
            visited.add(pos)
            visualize(maze, None, visited)  # Visualize the maze with visited nodes
            print(f"  Visited: {pos}")
            if pos == end_pos:
                # Found the end position, reconstruct the path
                path = [end_pos]
                while path[-1] != start_pos:
                    path.append(parent[path[-1]])
                path.reverse()
                visualize(maze, path)  # Visualize the maze with the path found
                return path
            visited.add(pos)
            for neighbor in get_neighbors(maze, pos):
                if neighbor not in visited:
                    stack.append(neighbor)
                    parent[neighbor] = pos
                    print(f"  Added to queue: {neighbor}")
        # End position is unreachable
        visualize(maze, None, visited)  # Visualize the maze with visited nodes
        return None

    # Load the maze from the file
    maze = load_maze()

    # Find the start and end positions in the maze
    start_pos, end_pos = find_start_end(maze)

    # Solve the maze using the specified search algorithm
    if search_algo == "bfs":
        path = bfs(maze, start_pos, end_pos)
    elif search_algo == "dfs":
        path = dfs(maze, start_pos, end_pos)
    else:
        print("Invalid search algorithm specified!")
        return

    # Visualize the maze and the path found by the algorithm
    print(" Final Solution:")
    visualize(maze, path)


if __name__ == '__main__':
    # Load the maze from the file
    maze = load_maze()

    # Print the loaded maze
    print("Maze:")
    print('\n'.join([''.join(row) for row in maze]))

    # Prompt the user for the search algorithm to use
    search_algo = input(
        "Enter 'bfs' to use breadth-first search or 'dfs' to use depth-first search: ")

    # Call the solve_maze function with the specified search algorithm
    solve_maze(search_algo.lower())
