"""
cartpole_GL_FLIF.py

This script implements and trains a Spiking Neural Network (SNN) with custom 
Fractional-Order Leaky Integrate-and-Fire (FLIF) neurons to solve the 
CartPole-v1 environment from the Gymnasium library.

The core of the network uses a fractional-order neuron model based on the 
Grünwald-Letnikov (GL) derivative to introduce memory effects into the neuron's
dynamics. The network is trained using the REINFORCE policy gradient algorithm.

The script will run a specified number of independent trials, with each trial
running for a maximum number of episodes. Performance, including the episode at
which the environment is solved, is tracked and printed.

Dependencies:
- numpy
- torch
- snntorch
- gymnasium

Author: [Matthew Tubbin/MTubbin]
Date: [8/7/2025]

Warning: IF USING MULTIPROCESSING ADD TRUE RANDOMNESS (RNG ISSUE)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from snntorch import surrogate
import gymnasium as gym
from collections import deque

# --- Setup Environment ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# --- Main Parameters/Hyperparameters ---
num_trials = 100 # trials to run in sequence
num_episodes = 3000 # total episodes for each trial
hidden_size = 16 # nodes / neurons in hidden layer
gamma = 0.99 # discount factor | larger = long term focus | 0.99 standard
learning_rate = 0.0014 
num_steps = 12  # SNN temporal window





# --- FLIF Neuron Equation ---
class FLIF(nn.Module):
    def __init__(self, size, alpha=0.7, tau_m=40.0, V_th=1.0, V_reset=0.0, dt=1.0, r=1.0):
        super(FLIF, self).__init__()
        self.alpha = alpha # fractional dynamics value (0.5 - 0.7) fastest, 0.1-0.2 LONG training window
        self.tau_m = tau_m # 40.0 gave best results, changed from 20.0
        self.V_th = V_th # membrane potential threshold
        self.V_reset = V_reset # membrane potential reset voltage
        self.dt = dt # time step, 1.0 standard
        self.num_neurons = size # hidden_size
        self.r_m = r # membrane resistance, 1.0 standard, no current effect

        self.register_buffer("gl_coeffs", self.binom_coeff(alpha, num_steps)) # history buffer same size as num_steps 
        self.spike_grad = surrogate.fast_sigmoid(slope=16) # surrogate gradient from snntorch for "dead neurons"

    def binom_coeff(self, alpha, length):
        # GL coefficient EQ.10 from numerical approximations paper (A.M. AbdelAty)
        coeffs = torch.zeros(length)
        coeffs[0] = 1.0
        for k in range(1, length):
            coeffs[k] = (1 - (alpha + 1) / k) * coeffs[k - 1]
        return coeffs # full of values of some magnitude with 1 being in the front and decreasing in magnitude as going further back to end num_steps -1 in length
        # [+1, -negative, -smaller_negative, -even_smaller_negative, ...]

    def init_state(self, batch_size):
        # Initialize voltage and a history of zeros
        voltage = torch.zeros(batch_size, self.num_neurons, device=device)
        voltage_history = torch.zeros(batch_size, self.num_neurons, num_steps, device=device)
        return voltage, voltage_history

    def forward(self, input_current, voltage, voltage_history):
        prev_v = self.alpha * voltage # previous voltage value, markovian, first term 3.1

        input_term = (self.dt ** self.alpha) * (((self.r_m*input_current) - voltage + self.V_reset)/self.tau_m) # Leak term, simplifies to LIF when alpha = 1.0
        # simplifies to (self.dt ** self.alpha) * (input_current - voltage / self.tau_m), second term 3.2

        num_hist_terms = len(self.gl_coeffs) - 2 # terms to include for memory sum
        hist_terms = voltage_history[:, :, 1:num_hist_terms+1] # shift history terms
        gl_terms = self.gl_coeffs[2:].view(1, 1, -1) # call binomial coefficients
        history_sum = (hist_terms * gl_terms).sum(dim=2) # history length of gl * voltage history, third term 3.3

        new_voltage = prev_v + input_term - history_sum # based off formula 39.31 from fractional order dynamics book (fidel)
        # 3.1 + 3.2 - 3.3 from EQ
        
        spike = self.spike_grad(new_voltage - self.V_th) # spike threshold and reset
        voltage_after_spike = new_voltage * (1.0 - spike) + self.V_reset * spike # if no spike voltage is same, if spike voltage reset to 0, spike = 1 or 0

        # The new voltage (after reset) becomes the most recent history item
        new_history = torch.roll(voltage_history, shifts=1, dims=2)
        new_history[:, :, 0] = voltage_after_spike # append voltage to newest value

        return spike, voltage_after_spike, new_history 





# --- SNN Policy Model ---
class SNN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(SNN, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size) # raw value input
        self.flif1 = FLIF(hidden_size) # send current into FLIF neuron
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.flif2 = FLIF(action_size)

    def forward(self, x, mem1, hist1, mem2, hist2):
        spk2_rec = [] # record spike count from action output
        cur1 = self.fc1(x)
        for _ in range(num_steps): # hold input current from state observation for num_steps time
            spk1, mem1, hist1 = self.flif1(cur1 * self.flif1.tau_m, mem1, hist1) # scale input current by tau_m making input direct 

            cur2 = self.fc2(spk1)
            spk2, mem2, hist2 = self.flif2(cur2 * self.flif2.tau_m, mem2, hist2) # scaled current same for output layer

            spk2_rec.append(spk2)

        out_spk_cnt = torch.stack(spk2_rec, dim=0).sum(dim=0) # spike count torch 
        return out_spk_cnt, mem1, hist1, mem2, hist2






# --- Data Collection and Main Loop ---
all_runs_durations = []
episode_solved = []

for run in range(num_trials):
    print(f"--- Starting Run {run + 1}/{num_trials} ---")
    
    policy = SNN(state_size, action_size, hidden_size).to(device) # initialize policy
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate) # initialize optimizer, Adam standard, RMSprop optional alternative
    
    durations_for_this_run = []
    duration_deque = deque(maxlen=100) # past 100 values, newest value kicks out oldest

    for episode in range(num_episodes):
        state, _ = env.reset() # gets state observation from cartpole env. (position, velocity, angle, angle velocity)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # turn into torch compatible
        
        # Initialize neuron states ONCE per episode, history reset each episode
        mem1, hist1 = policy.flif1.init_state(batch_size=1) 
        mem2, hist2 = policy.flif2.init_state(batch_size=1)
        
        done = False
        log_probs_saved, Rewards = [], []

        while not done:
            spike_count, mem1, hist1, mem2, hist2 = policy(state, mem1, hist1, mem2, hist2) # send observations and history into policy
            #print(f"Spike Counts: {spike_count.cpu().detach().numpy()}") # print spike count for observation, cluttered

            dist = Categorical(logits=spike_count) # get distribution of spike results from action output
            action = dist.sample() # choose action from distribution
            log_probs_saved.append(dist.log_prob(action)) # for future loss calculation, simplified if calculated now
 
            next_state, reward, term, trunc, _ = env.step(action.item()) # send action into environment and get resulting state observation, continues until termination
            done = term or trunc # term out of bounds, trunc max duration 500
            Rewards.append(reward) # track total reward count for episode
            state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0) # make current state the next state value, send to policy
        
        # --- REINFORCE update ---
        episode_duration = sum(Rewards)
        durations_for_this_run.append(episode_duration)
        duration_deque.append(episode_duration)

        G = 0 # G_t from EQ
        returns = []
        for r in reversed(Rewards):
            G = r + gamma * G 
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device) # make returns torch compatible
        returns = (returns - returns.mean()) / (returns.std() + 1e-9) # normalize for variance reduction

        policy_loss = []
        for log_prob, reward in zip(log_probs_saved, returns): # calculate policy loss
            policy_loss.append(-log_prob * G)

        optimizer.zero_grad() # clear gradients
        loss = torch.stack(policy_loss).sum() # sum loss 
        loss.backward() # compute gradients for current episode
        optimizer.step() # optimize weights and biases using gradients

        if episode % 100 == 0: # change to print more often/less
            print(f"Episode {episode} | Last Score: {episode_duration:.2f} | Avg Score (last 100): {np.mean(duration_deque):.2f}")

        if len(duration_deque) == 100 and np.mean(duration_deque) >= 475.0: # solving criteria
            print(f"Environment Solved at episode {episode}!")
            episode_solved.append(episode)
            remaining_eps = num_episodes - (episode + 1)
            if remaining_eps > 0:
                durations_for_this_run.extend([500.0] * remaining_eps)
            break
    
    while len(durations_for_this_run) < num_episodes:
        durations_for_this_run.append(durations_for_this_run[-1])

    all_runs_durations.append(durations_for_this_run)

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