import numpy as np
import matplotlib.pyplot as plt
import math

def plot_results(all_runs_durations, episode_solved, num_episodes, num_runs, agent):
    all_runs_durations = np.array(all_runs_durations)

    successful_runs = sum(1 for ep in episode_solved if ep < num_episodes)
    comp_percent = (successful_runs / num_runs) * 100
    print(f"\nCompletion Rate: {comp_percent:.2f}% of trials solved the environment within {num_episodes} episodes.")

    average_solve_episode = int(np.mean(episode_solved)) # y axis
    print(f"Average solve episode across runs: {average_solve_episode}")

    x_limit = int(math.ceil((average_solve_episode + 50) / 100.0)) * 100 # x limit with space at end
    mean_dur = np.mean(all_runs_durations, axis=0) # mean duration
    std_dur = np.std(all_runs_durations, axis=0) # standard deviation
    episodes = np.arange(num_episodes) # x axis

    # --- Plotting ---
    plt.figure(figsize=(12, 6)) 

    # Plot the mean duration
    plt.plot(episodes, mean_dur, label='Mean Episode Duration')

    # Create and plot the standard deviation fill
    up_bnd = mean_dur + std_dur
    low_bnd = mean_dur - std_dur
    max_score = 500
    up_bnd_max = np.clip(up_bnd, a_min=None, a_max=max_score)
    plt.fill_between(episodes, low_bnd, up_bnd_max, color='skyblue', alpha=0.3, label='Standard Deviation')

    # Plot reference lines
    # linestyle = (0, (5, 10)) : loosely dashed, potentially switch to loosely dotted
    plt.axhline(y=max_score, color='r', linestyle=(0, (5, 10)), linewidth=1.5, label='Max Duration (500)')
    plt.axvline(x=average_solve_episode, color='green', linestyle='solid', linewidth=2, label=f'Avg Solve Ep ({average_solve_episode})')

    # Add labels and title
    plt.xlim(0, x_limit)
    plt.title(f'{agent} Performance Across {num_runs} Trials ({comp_percent:.0f}% Completion Rate)')
    plt.xlabel('Episode')
    plt.ylabel('Duration (Total Reward)')
    plt.legend()
    plt.grid(True)
    plt.show()