import pickle
import numpy as np
import random
from llast import GameEnvironment
import matplotlib.pyplot as plt
import gzip
import os

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        total = len(self.buffer)
        prios = self.priorities[:total]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(total, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = np.min([1.0, self.beta + self.beta_increment])
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + self.epsilon

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0,replay_capacity=10000, batch_size=64):
        self.env = env

        # Q-table initialisation
        self.q_table = {}

        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Experience playback buffer
        self.replay_buffer = PrioritizedReplayBuffer(replay_capacity)
        self.batch_size = batch_size

        # Training records
        self.best_score = -np.inf
        self.scores = []

    def state_to_index(self, state):
        """Converting a continuous state to a discrete index"""
        return tuple(state)

    def get_action(self, state):
        """epsilon-greedy strategy selection action"""
        if np.random.random() < self.epsilon:
            return np.random.choice([-1, 0, 1, 2])  # Random explorations
        else:
            state_index = self.state_to_index(state)
            # If the state does not exist, initialise the Q value to 0
            if state_index not in self.q_table:
                self.q_table[state_index] = np.zeros(4)
            return np.argmax(self.q_table[state_index]) - 1  # Converted to real action

    def update_q_table(self, experiences, weights, indices):
        for i in range(len(experiences)):
            experience = experiences[i]
            weight = weights[i]
            idx = indices[i]

            state, action, reward, next_state, done = experience
            action_idx = action + 1  # Map actions to 0-3 indexes

            state_index = self.state_to_index(state)
            next_state_index = self.state_to_index(next_state)

            # Initialise Q values
            if state_index not in self.q_table:
                self.q_table[state_index] = np.zeros(4)
            if next_state_index not in self.q_table:
                self.q_table[next_state_index] = np.zeros(4)

            old_value = self.q_table[state_index][action_idx]
            next_max = np.max(self.q_table[next_state_index]) if not done else 0

            td_error = reward + self.gamma * next_max - old_value
            new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value) * weight
            self.q_table[state_index][action_idx] = new_value

            # Update priorities
            self.replay_buffer.update_priorities([idx], [abs(td_error)])

    def add_experience(self, experience):
        """Add experience to playback buffer"""
        self.replay_buffer.add(experience)

    def sample_experience(self,batch_size):
        """Random sampling of small batches of experience from the playback buffer"""
        return self.replay_buffer.sample(batch_size)

    def save_model(self, filename='best_q_agent_l.pkl'):
        """Save the training results with compression"""
        try:
            with gzip.open(filename, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'best_score': self.best_score
                }, f)
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filename='best_q_agent_l.pkl'):
        """Load the training results with decompression"""
        if os.path.getsize(filename) > 0:
            try:
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                    self.q_table = data['q_table']
                    self.best_score = data['best_score']
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"File {filename} is empty or corrupted")


def train_agent(episodes=1000):
    env = GameEnvironment()
    agent = QLearningAgent(env, replay_capacity=10000, batch_size=64)

    # Training parameters
    epsilon_decay = 0.995
    min_epsilon = 0.01

    score_history = []  # Save the score for each round
    reward_history = []  # Save total rewards per round

    for episode in range(episodes):
        state = env.get_state()
        total_reward = 0
        done = False
        episode_experiences = []

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, score = env.step(action)

            # Preservation of experience
            experience = (state, action, reward, next_state, done)
            episode_experiences.append(experience)

            # Add to priority experience playback buffer
            agent.add_experience(experience)

            # Frame-by-frame updating of the Q-table
            if len(agent.replay_buffer) >= agent.replay_buffer.capacity // 2:
                experiences, indices, weights = agent.sample_experience(agent.batch_size)
                agent.update_q_table(experiences, weights,indices)

            state = next_state
            total_reward += reward

            if done:
                break

        # Record the score and total_reward for the round.
        score_history.append(score)
        reward_history.append(total_reward)

        # Decay exploration rate
        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)

        # Document and save the best models
        agent.scores.append(score)
        if score > agent.best_score:
            agent.best_score = score
            agent.save_model()

        # Progress output
        if episode % 1 == 0:
            print(f"Episode: {episode}, Score: {score}, Total_reward:{total_reward}, Epsilon: {agent.epsilon:.2f}")
            if episode % 50 == 0:
                agent.save_model(f'q_agent_l_{episode}.pkl')

        # Save the final model
    agent.save_model('last_q_agent_l.pkl')

    # Mapping rewards and changes in scores
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(reward_history, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(score_history, label="Score")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Score per Episode")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics_l.png")
    plt.show()


    env.close()
    return agent

# Testing with trained intelligences
def test_agent():
    env = GameEnvironment()
    agent = QLearningAgent(env)
    agent.load_model()
    agent.epsilon = 0  # Close the quest

    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        state, reward, done, score = env.step(action)

    print("Final Score:", score)
    env.close()


if __name__ == "__main__":
    agent = train_agent(episodes=1000)
