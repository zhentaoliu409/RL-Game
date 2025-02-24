from llast import GameEnvironment
from llast_ai import QLearningAgent

def demo_agent():
    env = GameEnvironment()
    agent = QLearningAgent(env,epsilon=0)
    agent.epsilon = 0  # Set the exploration rate to 0, i.e., no exploration, selecting actions exactly according to the Q-table
    agent.load_model()  # Load the trained Q-table
    # print(f"agent.best_score: {agent.best_score}, agent.epsilon: {agent.epsilon}")

    episode = 1
    while True:
        episode += 1
        state = env.get_state()  # Reset the environment
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)  # Intelligent body selective action
            next_state, reward, done, score = env.step(action)  # Execute the action
            total_reward += reward
            state = next_state

        #print(f"Episode {episode} finished with Score: {score}, Total Reward: {total_reward}")

demo_agent()