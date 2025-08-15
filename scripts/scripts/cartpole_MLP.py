"""
cartpole_MLP_final.py

This script trains a standard Artificial Neural Network (ANN), specifically a 
Multi-Layer Perceptron (MLP), to solve the CartPole-v1 environment from 
the Gymnasium library.

The network is trained using the REINFORCE policy gradient algorithm. 
The script will run a specified number of independent trials and log the 
performance, including the average episode number required to solve the environment.

Dependencies:
- numpy
- torch
- gymnasium

Author: [Matthew Tubbin/MTubbin]
Date: [8/07/2025]

warning: IF USING WITH MULTIPROCESSING ADD TRUE RANDOMNESS
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical
from collections import deque


# --- Environment Setup ---
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# --- Main Parameters/Hyperparameters ---
num_trials = 100 # trials to run in sequence
num_episodes = 3000 # total episodes for each trial
hidden_size = 16 # nodes / neurons in hidden layer
gamma = 0.99 # discount factor | larger = long term focus | 0.99 standard
learning_rate = 0.001





# --- ANN Policy Model ---
class ANN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # ReLU activation function from paper
        x = self.fc2(x) # send current through output action
        return x






# --- Data Collection for All Runs ---
all_runs_durations = []
episode_solved = []

# --- Main Loop for Multiple Runs ---
for run in range(num_trials):
    print(f"--- Starting Run {run + 1}/{num_trials} ---")
    
    # reintialized for each trial
    policy = ANN(state_size, action_size, hidden_size) # initialize policy
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate) # initialize optimizer, Adam standard, RMSprop optional alternative
    
    durations_for_this_run = []
    duration_deque = deque(maxlen=100) # past 100 values, newest value kicks out oldest

    for episode in range(num_episodes):
        state, _ = env.reset() # gets state observation from cartpole env. (position, velocity, angle, angle velocity)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # turn into torch compatible
        done = False
        
        log_probs_saved, Rewards = [], []

        # Data Collection loop for a single episode
        while not done:
            logits = policy(state) # raw values befor activation function
            dist = Categorical(logits=logits) # distribution post activation function
            action = dist.sample() # choose action from dist
            log_probs_saved.append(dist.log_prob(action)) # calculate log probs earlier for simplification

            next_state, reward, term, trunc, _ = env.step(action.item()) # send action into policy and get next state
            done = term or trunc # term = out of bounds | trunc = max duration 500
            Rewards.append(reward) # append rewards from episode
            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) # current state becomes next state

        episode_duration = sum(Rewards)
        durations_for_this_run.append(episode_duration)
        duration_deque.append(episode_duration)

        # Discounted returns calculation
        DiscountedReturns = []
        G = 0 # G_t
        for r in reversed(Rewards):
            G = r + gamma * G
            DiscountedReturns.insert(0, G)
        
        DiscountedReturns = torch.tensor(DiscountedReturns)
        DiscountedReturns = (DiscountedReturns - DiscountedReturns.mean()) / (DiscountedReturns.std() + 1e-9) # Normalize returns to reduce variance

        # Policy loss calculation
        policy_loss = []
        for log_prob, G in zip(log_probs_saved, DiscountedReturns):
            policy_loss.append(-log_prob * G)

        optimizer.zero_grad() # clear gradients
        loss = torch.stack(policy_loss).sum() # sum losses
        loss.backward() # compute gradients for current episode
        optimizer.step() # optimize weights and biases using gradients

        if episode % 100 == 0: # vary episode print frequency
            print(f"Episode {episode} | Average Score (last 100): {np.mean(duration_deque):.2f}")

        if len(duration_deque) == 100 and np.mean(duration_deque) >= 475.0:
            print(f"Environment Solved at episode {episode}!")
            episode_solved.append(episode)
            # Fill remaining episodes with the solved score to keep data arrays consistent
            remaining_eps = num_episodes - (episode + 1)
            if remaining_eps > 0:
                durations_for_this_run.extend([500.0] * remaining_eps)
            break
    
    # Ensure each run has the same number of episodes for consistent array shapes
    while len(durations_for_this_run) < num_episodes:
        durations_for_this_run.append(durations_for_this_run[-1]) # Pad with the last score

    all_runs_durations.append(durations_for_this_run)

# Pad any runs that never solved
while len(episode_solved) < num_trials:
    episode_solved.append(num_episodes)

env.close()




print("\n--- Training Complete ---")
print(f"Average solve episode: {np.mean(episode_solved):.2f}")

mean_solve_ep = np.mean(episode_solved)
print(f"Avg solve ep: {mean_solve_ep:.2f}")
all_durations_np = np.array(all_runs_durations) # Convert the list of lists into a 2D NumPy array
std_dev_per_episode = np.std(all_durations_np, axis=0) # Calculate the standard deviation for each episode across all runs
avg_std_dev = np.mean(std_dev_per_episode) # Calculate the average of these standard deviations
print(f"Average standard deviation across all trials: {avg_std_dev:.2f}")
