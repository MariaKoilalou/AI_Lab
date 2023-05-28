import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def discretize_state(state):
    """
    This function takes a state (continuous) and returns a discretized representation of the state.
    """
    bins = [
        np.linspace(-4.8, 4.8, 24),
        np.linspace(-5, 5, 24),
        np.linspace(-0.418, 0.418, 48),
        np.linspace(-5, 5, 24)
    ]

    state_index = []
    for i in range(len(state[0])):
        state_index.append(np.digitize(state[0][i], bins[i]) - 1)

    return tuple(state_index)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    n_states = (24, 24, 48, 48)

    alpha = 0.7  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Start with exploration
    epsilon_min = 0.01  # Maintain some exploration
    epsilon_decay = 0.995  # Decay rate of epsilon
    n_episodes = 1000
    max_steps = 100

    rewards_per_episode = []

    Q = np.zeros(n_states + (n_actions,))

    for episode in range(n_episodes):
        state = discretize_state(env.reset())
        total_reward = 0

        for step in range(max_steps):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state((next_state, info))

            current_q = Q[state + (action,)]
            max_next_q = np.max(Q[next_state])
            new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)

            Q[state + (action,)] = new_q

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        rewards_per_episode.append(total_reward)

        print(f"Episode: {episode + 1}, Steps: {step + 1}, Total Reward: {total_reward}")
        print(f"Max Q-Value: {np.max(Q)}, Sum of Q-Values: {np.sum(Q)}")

        # decay epsilon
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

    state = discretize_state(env.reset())
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = np.argmax(Q[state])
        next_state, _, terminated, truncated, _ = env.step(action)
        state = discretize_state((next_state, {}))

    env.close()

