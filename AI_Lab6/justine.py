import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Create the gymnasium environment
    env = gym.make('CartPole-v1')

    # Get the number of states and actions
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Hyperparameters for the Q-learning algorithm
    alpha = 0.7  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 0.5  # Exploration rate
    learning_rate = 0.1
    discount_factor = 0.99
    n_episodes = 1000
    max_steps = 100

    # List to store rewards per episode
    rewards_per_episode = []

    # Initialize the Q-table
    Q = np.zeros((n_states, n_actions))

    # Main loop for episodes
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # Choose action from the current state with an epsilon-greedy policy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                state_array = state[0].astype(int)
                action = np.argmax(Q[state_array])

            # Execute the action and observe the new state, reward, terminated, truncated, info
            next_state, reward, terminated, truncated, info = env.step(action)

            # Update the Q-value for the current state-action pair
            current_q = Q[state_array, action]
            next_state_array = next_state[0].astype(int)
            max_next_q = np.max(Q[next_state_array])
            new_q = current_q + alpha * (reward + gamma*max_next_q-current_q)

            Q[state_array, action] = new_q

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        rewards_per_episode.append(total_reward)

        print("Q-table:")
        print(Q)

        print(
            f"Episode: {episode + 1}, Steps: {step + 1}, Total Reward: {total_reward}")

    # Use the learned Q-table to choose the best actions in the environment
    state = env.reset()
    done = False

    # Plotting rewards per episode
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

    while not done:
        state_array = state[0].astype(int)
        action = np.argmax(Q[state_array])
        next_state, _, terminated, truncated, _ = env.step(action)
        state = next_state

        if terminated or truncated:
            break

    env.close()
