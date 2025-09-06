"""
Reinforcement Learning Example: Q-Learning and Policy Gradient

This example demonstrates:
1. Q-Learning for a simple grid world environment
2. Policy Gradient (REINFORCE) for CartPole environment
3. Comparison of different RL algorithms

We'll create a custom grid world and use OpenAI Gym's CartPole environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
from collections import defaultdict, deque
import random

class GridWorld:
    """Simple grid world environment for Q-Learning demonstration"""
    
    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.obstacles = [(1, 1), (2, 2), (3, 1)]  # Some obstacles
        self.current_pos = self.start
        self.actions = ['up', 'down', 'left', 'right']
        self.action_map = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
    
    def reset(self):
        """Reset the environment to initial state"""
        self.current_pos = self.start
        return self.current_pos
    
    def step(self, action):
        """Take a step in the environment"""
        dx, dy = self.action_map[action]
        new_x = max(0, min(self.size-1, self.current_pos[0] + dx))
        new_y = max(0, min(self.size-1, self.current_pos[1] + dy))
        new_pos = (new_x, new_y)
        
        # Check for obstacles
        if new_pos in self.obstacles:
            reward = -1
            done = False
        else:
            self.current_pos = new_pos
            if new_pos == self.goal:
                reward = 10
                done = True
            else:
                reward = -0.1  # Small negative reward for each step
                done = False
        
        return self.current_pos, reward, done
    
    def render(self):
        """Visualize the grid world"""
        grid = np.zeros((self.size, self.size))
        
        # Mark obstacles
        for obs in self.obstacles:
            grid[obs] = -1
        
        # Mark goal
        grid[self.goal] = 2
        
        # Mark current position
        grid[self.current_pos] = 1
        
        return grid

class QLearningAgent:
    """Q-Learning agent for the grid world"""
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(len(env.actions)))
        self.episode_rewards = []
        self.episode_lengths = []
    
    def choose_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.choice(self.env.actions)
        else:
            return self.env.actions[np.argmax(self.q_table[state])]
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule"""
        action_idx = self.env.actions.index(action)
        current_q = self.q_table[state][action_idx]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action_idx] = new_q
    
    def train(self, num_episodes=1000):
        """Train the Q-learning agent"""
        print("Training Q-Learning Agent...")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            max_steps = 100
            
            while steps < max_steps:
                action = self.choose_action(state, training=True)
                next_state, reward, done = self.env.step(action)
                
                self.update_q_table(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            # Decay epsilon
            if episode % 100 == 0:
                self.epsilon = max(0.01, self.epsilon * 0.99)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    def test(self, num_episodes=10):
        """Test the trained agent"""
        print("\nTesting Q-Learning Agent...")
        test_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            max_steps = 100
            
            while steps < max_steps:
                action = self.choose_action(state, training=False)
                next_state, reward, done = self.env.step(action)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            test_rewards.append(total_reward)
            print(f"Test Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
        
        print(f"Average Test Reward: {np.mean(test_rewards):.2f}")
        return test_rewards

class PolicyGradientAgent:
    """Policy Gradient (REINFORCE) agent for CartPole"""
    
    def __init__(self, env, learning_rate=0.01, discount_factor=0.99):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        
        # Simple neural network parameters (linear policy)
        self.weights = np.random.normal(0, 0.1, (env.observation_space.shape[0], env.action_space.n))
        self.episode_rewards = []
        self.episode_lengths = []
    
    def policy(self, state):
        """Compute action probabilities using softmax policy"""
        logits = np.dot(state, self.weights)
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        return probabilities
    
    def choose_action(self, state):
        """Sample action from policy"""
        probabilities = self.policy(state)
        action = np.random.choice(self.env.action_space.n, p=probabilities)
        return action, probabilities
    
    def compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = []
        discounted_sum = 0
        for reward in reversed(rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        return returns
    
    def update_policy(self, states, actions, returns):
        """Update policy using REINFORCE algorithm"""
        for state, action, return_val in zip(states, actions, returns):
            probabilities = self.policy(state)
            action_prob = probabilities[action]
            
            # Compute gradient
            gradient = np.zeros_like(self.weights)
            for a in range(self.env.action_space.n):
                if a == action:
                    gradient[:, a] = state * (1 - action_prob)
                else:
                    gradient[:, a] = -state * probabilities[a]
            
            # Update weights
            self.weights += self.lr * return_val * gradient
    
    def train(self, num_episodes=1000):
        """Train the policy gradient agent"""
        print("Training Policy Gradient Agent...")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            states, actions, rewards = [], [], []
            total_reward = 0
            steps = 0
            max_steps = 500
            
            while steps < max_steps:
                action, _ = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Compute returns and update policy
            returns = self.compute_returns(rewards)
            self.update_policy(states, actions, returns)
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    def test(self, num_episodes=10):
        """Test the trained agent"""
        print("\nTesting Policy Gradient Agent...")
        test_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            max_steps = 500
            
            while steps < max_steps:
                action, _ = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            test_rewards.append(total_reward)
            print(f"Test Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
        
        print(f"Average Test Reward: {np.mean(test_rewards):.2f}")
        return test_rewards

def q_learning_example():
    """Demonstrate Q-Learning on grid world"""
    print("=" * 50)
    print("REINFORCEMENT LEARNING: Q-LEARNING")
    print("=" * 50)
    
    # Create environment and agent
    env = GridWorld(size=5)
    agent = QLearningAgent(env)
    
    # Train the agent
    agent.train(num_episodes=1000)
    
    # Test the agent
    test_rewards = agent.test(num_episodes=5)
    
    # Visualize training progress
    plt.figure(figsize=(15, 5))
    
    # Plot rewards
    plt.subplot(1, 3, 1)
    plt.plot(agent.episode_rewards)
    plt.title('Q-Learning: Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Plot episode lengths
    plt.subplot(1, 3, 2)
    plt.plot(agent.episode_lengths)
    plt.title('Q-Learning: Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    # Visualize Q-table (for first few states)
    plt.subplot(1, 3, 3)
    states = list(agent.q_table.keys())[:10]  # First 10 states
    q_values = [np.max(agent.q_table[state]) for state in states]
    plt.bar(range(len(states)), q_values)
    plt.title('Q-Values for First 10 States')
    plt.xlabel('State Index')
    plt.ylabel('Max Q-Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/vignesh/Documents/GitHub/Generative AI/ml_examples/q_learning_results.png')
    plt.show()
    
    return agent, test_rewards

def policy_gradient_example():
    """Demonstrate Policy Gradient on CartPole"""
    print("\n" + "=" * 50)
    print("REINFORCEMENT LEARNING: POLICY GRADIENT")
    print("=" * 50)
    
    # Create environment and agent
    env = gym.make('CartPole-v1')
    agent = PolicyGradientAgent(env)
    
    # Train the agent
    agent.train(num_episodes=1000)
    
    # Test the agent
    test_rewards = agent.test(num_episodes=5)
    
    # Visualize training progress
    plt.figure(figsize=(15, 5))
    
    # Plot rewards
    plt.subplot(1, 3, 1)
    plt.plot(agent.episode_rewards)
    plt.title('Policy Gradient: Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Plot episode lengths
    plt.subplot(1, 3, 2)
    plt.plot(agent.episode_lengths)
    plt.title('Policy Gradient: Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    # Plot moving average
    plt.subplot(1, 3, 3)
    window = 50
    moving_avg = np.convolve(agent.episode_rewards, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg)
    plt.title(f'Policy Gradient: Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/vignesh/Documents/GitHub/Generative AI/ml_examples/policy_gradient_results.png')
    plt.show()
    
    env.close()
    return agent, test_rewards

def compare_algorithms():
    """Compare different RL algorithms"""
    print("\n" + "=" * 50)
    print("ALGORITHM COMPARISON")
    print("=" * 50)
    
    # Q-Learning results
    env = GridWorld(size=5)
    q_agent = QLearningAgent(env, epsilon=0.1)
    q_agent.train(num_episodes=500)
    q_test_rewards = q_agent.test(num_episodes=10)
    
    # Q-Learning with different epsilon
    q_agent_greedy = QLearningAgent(env, epsilon=0.01)
    q_agent_greedy.train(num_episodes=500)
    q_greedy_test_rewards = q_agent_greedy.test(num_episodes=10)
    
    # Compare results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(q_agent.episode_rewards, label='Q-Learning (ε=0.1)', alpha=0.7)
    plt.plot(q_agent_greedy.episode_rewards, label='Q-Learning (ε=0.01)', alpha=0.7)
    plt.title('Q-Learning: Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    algorithms = ['Q-Learning\n(ε=0.1)', 'Q-Learning\n(ε=0.01)']
    avg_rewards = [np.mean(q_test_rewards), np.mean(q_greedy_test_rewards)]
    plt.bar(algorithms, avg_rewards)
    plt.title('Average Test Rewards')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/vignesh/Documents/GitHub/Generative AI/ml_examples/rl_algorithm_comparison.png')
    plt.show()
    
    print(f"Q-Learning (ε=0.1) Average Test Reward: {np.mean(q_test_rewards):.2f}")
    print(f"Q-Learning (ε=0.01) Average Test Reward: {np.mean(q_greedy_test_rewards):.2f}")

def main():
    """Main function to run all reinforcement learning examples"""
    print("REINFORCEMENT LEARNING EXAMPLES")
    print("This example demonstrates Q-Learning and Policy Gradient algorithms")
    print("on different environments.\n")
    
    # Run Q-Learning example
    q_agent, q_test_rewards = q_learning_example()
    
    # Run Policy Gradient example
    pg_agent, pg_test_rewards = policy_gradient_example()
    
    # Compare algorithms
    compare_algorithms()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("Reinforcement learning learns through interaction with an environment.")
    print("Key concepts demonstrated:")
    print("1. Q-Learning: Learns action-value function using temporal difference")
    print("2. Policy Gradient: Directly optimizes policy using policy gradient theorem")
    print("3. Exploration vs Exploitation: Balancing learning and performance")
    print(f"\nQ-Learning average test reward: {np.mean(q_test_rewards):.2f}")
    print(f"Policy Gradient average test reward: {np.mean(pg_test_rewards):.2f}")

if __name__ == "__main__":
    main()
