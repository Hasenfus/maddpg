import os
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def get_directories(base_path):
    """Get a list of all directories in the base path."""
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

def filter_and_sort_directories_by_date(directories, date_format):
    """Filter out directories that match the date and time pattern and sort them."""
    filtered_and_sorted_directories = []
    for directory in directories:
        try:
            # Parse the directory name into a datetime object
            date = datetime.strptime(directory, date_format)
            filtered_and_sorted_directories.append((directory, date))
        except ValueError:
            continue
    # Sort directories by date
    filtered_and_sorted_directories.sort(key=lambda x: x[1])
    return [directory for directory, date in filtered_and_sorted_directories]

def get_rewards_for_last_n_runs(base_path, n, date_format='%Y%m%d-%H%M%S', aggrew=True, valid=True, time=False, malfunction_agent = None):
    """Get the rewards for the last n runs."""
    directories = get_directories(base_path)
    date_directories = filter_and_sort_directories_by_date(directories, date_format)

    # Select the last n directories
    recent_n_directories = date_directories[-n:]

    all_rewards = []
    for directory in recent_n_directories:
        full_path = os.path.join(base_path, directory)
        rewards_data = []
        if malfunction_agent is not None:
            with open(os.path.join(full_path, 'test_'+  malfunction_agent + 'rewards.pkl'), 'rb') as f:
                rewards = pickle.load(f)
                rewards_data.append(rewards)

            if aggrew:
                with open(os.path.join(full_path, 'test_'+  malfunction_agent + '_agrewards.pkl'), 'rb') as f:
                    agrewards = pickle.load(f)
                    rewards_data.append(agrewards)
            else:
                rewards_data.append(None)

            if time:
                with open(os.path.join(full_path, 'test_'+  malfunction_agent + '_timesteps.pkl'), 'rb') as f:
                    time = pickle.load(f)
                    rewards_data.append(time)
            else:
                rewards_data.append(None)
        else:
            with open(os.path.join(full_path, 'test_rewards.pkl'), 'rb') as f:
                rewards = pickle.load(f)
                rewards_data.append(rewards)

            if aggrew:
                with open(os.path.join(full_path, 'test_agrewards.pkl'), 'rb') as f:
                    agrewards = pickle.load(f)
                    rewards_data.append(agrewards)
            else:
                rewards_data.append(None)

            if time:
                with open(os.path.join(full_path, 'test_timesteps.pkl'), 'rb') as f:
                    time = pickle.load(f)
                    rewards_data.append(time)
            else:
                rewards_data.append(None)

        all_rewards.append(tuple(rewards_data))

    return all_rewards

def calculate_mean_and_confidence_interval(data):
    """
    Calculate the mean and 95% confidence interval for each timestep.

    :param data: A list of lists, where each inner list represents a run and contains values for each timestep.
    :return: A tuple of two numpy arrays - one for the mean and one for the 95% confidence interval.
    """
    data = np.array(data)
    mean = np.mean(data, axis=0)
    stderr = stats.sem(data, axis=0, nan_policy='omit')
    confidence_interval = stderr * stats.t.ppf((1 + 0.95) / 2., len(data) - 1)

    return mean, confidence_interval


def average_and_confidence(run_info):
    """
    Calculate the average and 95% confidence interval for rewards at each timestep.

    :param output of get_rewards_for_last_n_runs
    :return: A tuple of four numpy arrays - mean rewards, confidence interval for rewards, mean timesteps, confidence interval for timesteps.
    """
    rewards = []
    timesteps = []
    for rewards_data, agrewards_data, time_data in run_info:
        rewards.append(rewards_data)
        # timesteps.append(time_data)
    mean_rewards, conf_rewards = calculate_mean_and_confidence_interval(rewards)
    # mean_timesteps, conf_timesteps = calculate_mean_and_confidence_interval(timesteps)

    return mean_rewards, conf_rewards #, mean_timesteps, conf_timesteps



# def plot_with_confidence_interval(mean_values, confidence_interval, timesteps, title="Plot with Confidence Interval", xlabel="Timestep", ylabel="Value",
#                                   ylim = None, save=False, save_path='/Users/Hunter/Development/Academic/UML/RL/Hasenfus-RL/Multi-Agent/maddpg/experiments/plots'):
#     """
#     Plot mean values with confidence interval using matplotlib.
#
#     :param mean_values: Array of mean values.
#     :param confidence_interval: Array of confidence interval values.
#     :param timesteps: Array of timesteps.
#     :param title: Title of the plot.
#     :param xlabel: Label for the x-axis.
#     :param ylabel: Label for the y-axis.
#     """
#
#     upper_bound = mean_values + confidence_interval
#     lower_bound = mean_values - confidence_interval
#
#     plt.figure(figsize=(4, 3))
#     plt.rcParams.update({
#         'font.size': 12,
#         'lines.linewidth': 2,
#         'axes.labelsize': 12,  # Axis label size
#         'axes.titlesize': 14,  # Title size
#         'figure.autolayout': True,  # Enable automatic layout adjustment
#     })
#
#     plt.plot(timesteps, mean_values, label="Mean", color="blue")
#     plt.fill_between(timesteps, lower_bound, upper_bound, color="blue", alpha=0.2, label="95% Confidence Interval")
#
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     if ylim:
#         plt.ylim(ylim)
#
#     plt.legend()
#     if save:
#         plt.savefig(os.path.join(save_path, title + '.png'), dpi=300)
#     plt.show()

# def plot_multiple_with_confidence_intervals(mean_values_list, confidence_intervals_list, timesteps, labels, title="Comparison Plot", xlabel="Timestep", ylabel="Value", save=False,
#                                             save_path='/Users/Hunter/Development/Academic/UML/RL/Hasenfus-RL/Multi-Agent/maddpg/experiments/plots', ylim=None):
#     """
#     Plot multiple sets of mean values with their confidence intervals.
#
#     :param mean_values_list: List of arrays of mean values for each algorithm.
#     :param confidence_intervals_list: List of arrays of confidence intervals for each algorithm.
#     :param timesteps: Array of timesteps.
#     :param labels: List of labels for each algorithm.
#     :param title: Title of the plot.
#     :param xlabel: Label for the x-axis.
#     :param ylabel: Label for the y-axis.
#     """
#     plt.figure(figsize=(10, 6))
#
#     for mean_values, confidence_interval, label in zip(mean_values_list, confidence_intervals_list, labels):
#         upper_bound = mean_values + confidence_interval
#         lower_bound = mean_values - confidence_interval
#
#         plt.plot(timesteps, mean_values, label=f"Mean - {label}")
#         plt.fill_between(timesteps, lower_bound, upper_bound, alpha=0.2, label=f"95% CI - {label}")
#
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     if ylim:
#         plt.ylim(ylim)
#
#     plt.legend()
#     if save:
#         plt.savefig(os.path.join(save_path, title+'.png'))
#     plt.show()
#
def plot_trajectories(trajectories, title="Agent Trajectories", xlabel="X Position", ylabel="Y Position",
                      save=False, save_path='/Users/Hunter/Development/Academic/UML/RL/Hasenfus-RL/Multi-Agent/maddpg/experiments/plots',  xlim=None, ylim=None, show_title= True):
    """
    Plot a series of x, y pairs as trajectories on an xy-plane.

    :param trajectories: A list of trajectories, where each trajectory is a list of (x, y) pairs.
    """
    fig, ax = plt.subplots(figsize=(4, 3))

    # Setting the plot's rcParams
    plt.rcParams.update({
        'font.size': 12,
        'lines.linewidth': 2,
        'axes.labelsize': 12,  # Axis label size
        'axes.titlesize': 14,  # Title size
        'figure.autolayout': True,  # Enable automatic layout adjustment
    })

    # Plotting each trajectory
    for i, traj in enumerate(trajectories):
        # Assuming each trajectory is a list of (x, y) tuples
        x_coords, y_coords = zip(*traj)
        ax.plot(x_coords, y_coords, 'r-')

    # Setting titles and labels
    if show_title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Adding grid
    ax.grid(True)

    # Remove top and right spines to get rid of the bounding box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Optionally, make left and bottom spines less obtrusive
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')

    # Adjusting plot limits if specified
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Save the figure if 'save' is True
    if save:
        fig.savefig(os.path.join(save_path, title + '.png'), dpi=300)

    # Show the plot
    plt.show()


def smooth_data(data, smoothing_factor=0.0):
    """
    Apply simple moving average smoothing to the data.
    :param data: Array of data points.
    :param smoothing_factor: Between 0 (no smoothing) and 1 (maximum smoothing).
    :return: Smoothed data.
    """
    if smoothing_factor <= 0:
        return data  # No smoothing
    window_size = int(len(data) * smoothing_factor) or 1
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_with_confidence_interval(mean_values, confidence_interval, timesteps, title="Plot with Confidence Interval", xlabel="Timestep", ylabel="Value",
                                  ylim=None, smoothing_factor=0.0, save=False, save_path='/Users/Hunter/Development/Academic/UML/RL/Hasenfus-RL/Multi-Agent/maddpg/experiments/plots', legend=True, show_title=True, color=None):
    """
    Plot mean values with confidence interval, with optional smoothing.

    :param mean_values: Array of mean values.
    :param confidence_interval: Array of confidence interval values.
    :param timesteps: Array of timesteps.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param smoothing_factor: Smoothing factor between 0 and 1.
    """
    smoothed_means = smooth_data(mean_values, smoothing_factor)
    smoothed_ci = smooth_data(confidence_interval, smoothing_factor)

    upper_bound = smoothed_means + smoothed_ci
    lower_bound = smoothed_means - smoothed_ci

    fig, ax = plt.subplots(figsize=(4, 3))

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make left and bottom spines darker
    ax.spines['left'].set_color('black')  # You can specify a hex code for colors as well, e.g., '#000000'
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)

    # Add grid behind the plot
    ax.grid(True, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    # Send grid to the back
    ax.set_axisbelow(True)

    # Update the rcParams for this figure specifically
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_size(14)

    # Plot data
    if color:
        ax.plot(timesteps[-len(smoothed_means):], smoothed_means, label="Mean", color=color)
        ax.fill_between(timesteps[-len(smoothed_means):], lower_bound, upper_bound, color=color, alpha=0.2,
                    label="95% Confidence Interval")
    else:
        ax.plot(timesteps[-len(smoothed_means):], smoothed_means, label="Mean", color="blue")
        ax.fill_between(timesteps[-len(smoothed_means):], lower_bound, upper_bound, color="blue", alpha=0.2,
                    label="95% Confidence Interval")

    # Setting titles and labels
    if show_title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)

    # Display legend
    if legend:
        ax.legend()

    # Save the figure if required
    if save:
        fig.savefig(os.path.join(save_path, title + '.png'), dpi=300)

    # Show the plot
    plt.show()
def plot_multiple_with_confidence_intervals(mean_values_list, confidence_intervals_list, timesteps, labels, title="Comparison Plot", xlabel="Timestep", ylabel="Value",
                                            save=False, legend = True,  save_path='/Users/Hunter/Development/Academic/UML/RL/Hasenfus-RL/Multi-Agent/maddpg/experiments/plots', ylim=None, smoothing_factor=0.0, show_title=True, colors=None):
    """
    Plot multiple sets of mean values with their confidence intervals, with optional smoothing.

    :param mean_values_list: List of arrays of mean values for each algorithm.
    :param confidence_intervals_list: List of arrays of confidence intervals for each algorithm.
    :param timesteps: Array of timesteps.
    :param labels: List of labels for each algorithm.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param smoothing_factor: Smoothing factor between 0 and 1.
    """
    fig, ax = plt.subplots(figsize=(4, 3))

    # Update the rcParams for this figure specifically
    plt.rcParams.update({
        'font.size': 12,
        'lines.linewidth': 2,
        'axes.labelsize': 12,  # Axis label size
        'axes.titlesize': 14,  # Title size
        'figure.autolayout': True,  # Enable automatic layout adjustment
    })

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make left and bottom spines darker
    ax.spines['left'].set_color('black')  # You can specify a hex code for colors as well, e.g., '#000000'
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)

    # Add grid behind the plot
    ax.grid(True, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    # Send grid to the back
    ax.set_axisbelow(True)

    # Update the rcParams for this figure specifically
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_size(14)

    # Plotting mean values with confidence intervals
    for mean_values, confidence_interval, label in zip(mean_values_list, confidence_intervals_list, labels):
        smoothed_means = smooth_data(mean_values, smoothing_factor)
        smoothed_ci = smooth_data(confidence_interval, smoothing_factor)

        upper_bound = smoothed_means + smoothed_ci
        lower_bound = smoothed_means - smoothed_ci
        if colors:
            ax.plot(timesteps[-len(smoothed_means):], smoothed_means, label=f"Mean - {label}", color=colors[labels.index(label)])
            ax.fill_between(timesteps[-len(smoothed_means):], lower_bound, upper_bound, alpha=0.2,
                        label=f"95% CI - {label}", color=colors[labels.index(label)])
        else:
            ax.plot(timesteps[-len(smoothed_means):], smoothed_means, label=f"Mean - {label}")
            ax.fill_between(timesteps[-len(smoothed_means):], lower_bound, upper_bound, alpha=0.2,
                        label=f"95% CI - {label}")

    # Setting titles and labels
    if show_title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Adjusting plot limits if specified
    if ylim:
        ax.set_ylim(ylim)

    # Adding legend if specified
    if legend:
        ax.legend()

    # Save the figure if 'save' is True
    if save:
        fig.savefig(os.path.join(save_path, title + '.png'), dpi=300)

    # Show the plot
    plt.show()


def get_trajectories_and_distances(base_path, healthy = True, mal = False, distances = True, date_format='%Y%m%d-%H%M%S',):
    """Get the rewards for the last n runs."""
    directories = get_directories(base_path)
    date_directories = filter_and_sort_directories_by_date(directories, date_format)

    # recent_directory = date_directories[-1]
    #
    # full_path = os.path.join(base_path, recent_directory)
    # print(full_path)
    directories2 = get_directories(base_path)
    # print(directories2)
    date_directories2 = filter_and_sort_directories_by_date(directories2, date_format)
    # print(date_directories2)
    recent_directory2 = date_directories2[-1]

    full_path2 = os.path.join(base_path, recent_directory2)

    returnable = []
    if healthy:
        with open(os.path.join(full_path2, 'test_healthy_trajectories.pkl'), 'rb') as f:
            healthy_trajectories = pickle.load(f)
            returnable.append(healthy_trajectories)
    else:
        returnable.append(None)
    if mal:
        with open(os.path.join(full_path2, 'test_mal_trajectories.pkl'), 'rb') as f:
            mal_trajectories = pickle.load(f)
            returnable.append(mal_trajectories)
    else:
        returnable.append(None)
    if distances:
        with open(os.path.join(full_path2, 'test_healthy_distances.pkl'), 'rb') as f:
            distances = pickle.load(f)
            returnable.append(distances)
    else:
        returnable.append(None)

    return returnable


def plot_distance_distribution(distances, interval_width, save=False,title='alg1', save_path='/Users/Hunter/Development/Academic/UML/RL/Hasenfus-RL/Multi-Agent/maddpg/experiments/plots', show_title=True):
    """
    Plot the distribution of distances as a bar graph with specified interval widths.

    :param distances: A NumPy array of distances.
    :param interval_width: The width of each interval (e.g., 5 for 15-20, 20-25, etc.).
    """
    # Determine the range of distances
    min_distance = np.min(distances)
    max_distance = np.max(distances)

    # Create intervals
    bins = np.arange(min_distance, max_distance + interval_width, interval_width)

    # Calculate the histogram
    counts, _ = np.histogram(distances, bins=bins)

    # Calculate percentages
    percentages = (counts / counts.sum()) * 100

    # Define labels for the x-axis
    labels = [f'{int(bins[i])}-{int(bins[i + 1])}' for i in range(len(bins) - 1)]

    # Plotting
    plt.figure(figsize=(4, 3))

    plt.rcParams.update({
        'font.size': 12,
        'lines.linewidth': 2,
        'axes.labelsize': 12,  # Axis label size
        'axes.titlesize': 14,  # Title size
        'figure.autolayout': True,  # Enable automatic layout adjustment
    })

    plt.bar(labels, percentages, width=0.8, color='skyblue', edgecolor='black')

    if show_title:
        plt.title(title)
    plt.xlabel('Distance Intervals')
    plt.ylabel('Percentage of Runs (%)')

    plt.xticks(rotation=45)  # Rotate labels to improve readability
    plt.grid(axis='y', linestyle='--')
    if save:
        plt.savefig(os.path.join(save_path, title + 'distances.png'), dpi=300)
    plt.show()



def get_reward_function_breakdown(base_path, date_format='%Y%m%d-%H%M%S',):
    """Get the rewards for the last n runs."""
    directories = get_directories(base_path)
    date_directories = filter_and_sort_directories_by_date(directories, date_format)

    # recent_directory = date_directories[-1]
    #
    # full_path = os.path.join(base_path, recent_directory)
    # print(full_path)
    directories2 = get_directories(base_path)
    # print(directories2)
    date_directories2 = filter_and_sort_directories_by_date(directories2, date_format)
    # print(date_directories2)
    recent_directory2 = date_directories2[-1]

    full_path2 = os.path.join(base_path, recent_directory2)

    returnable = []
    with open(os.path.join(full_path2, 'test_healthy_rewards.pkl'), 'rb') as f:
        healthy_trajectories = pickle.load(f)
        returnable.append(healthy_trajectories)

    with open(os.path.join(full_path2, 'test_reward_ctrl.pkl'), 'rb') as f:
        mal_trajectories = pickle.load(f)
        returnable.append(mal_trajectories)

    with open(os.path.join(full_path2, 'test_reward_forward.pkl'), 'rb') as f:
        mal_trajectories = pickle.load(f)
        returnable.append(mal_trajectories)

    with open(os.path.join(full_path2, 'test_reward_survive.pkl'), 'rb') as f:
        mal_trajectories = pickle.load(f)
        returnable.append(mal_trajectories)

    with open(os.path.join(full_path2, 'test_reward_contact.pkl'), 'rb') as f:
        mal_trajectories = pickle.load(f)
        returnable.append(mal_trajectories)



    return returnable

def stacked_bar_graph(*tuples, bar_labels=None, array_labels=None, title="Stacked Bar Graph of Array Averages",
                      legend_size=(2, 2), legend_aspect='auto', show_title = False, colors=None, save = False, save_path='/Users/Hunter/Development/Academic/UML/RL/Hasenfus-RL/Multi-Agent/maddpg/experiments/plots'):
    averages = []

    assert len(tuples[0]) == len(array_labels)

    # Compute averages for each array in the tuples
    for i, arrays in enumerate(tuples):
        tuple_averages = []
        for arr in arrays:
            avg = np.mean(arr)
            tuple_averages.append(avg)
        averages.append(tuple_averages)

    # Set up the plot
    fig, ax = plt.subplots()
    bar_width = 0.8 / len(tuples)
    x = np.arange(len(tuples))

    # Create stacked bars for each tuple
    bottom_pos = np.zeros(len(tuples))
    bottom_neg = np.zeros(len(tuples))
    for i in range(len(averages[0])):
        values = [tuple_averages[i] for tuple_averages in averages]
        pos_mask = [val >= 0 for val in values]
        neg_mask = [val < 0 for val in values]

        # Plot positive rewards above the x-axis
        ax.bar(x[pos_mask], [val for val, mask in zip(values, pos_mask) if mask],
               bar_width, bottom=bottom_pos[pos_mask], label=array_labels[i] + " (Positive)",
               color=colors[i % len(colors)] if colors else None)

        # Plot negative rewards below the x-axis
        ax.bar(x[neg_mask], [val for val, mask in zip(values, neg_mask) if mask],
               bar_width, bottom=bottom_neg[neg_mask], label=array_labels[i] + " (Negative)",
               color=colors[i % len(colors)] if colors else None)

        # Display the average value on top of each stacked segment
        for j, value in enumerate(values):
            if value >= 0:
                ax.text(x[j], bottom_pos[j] + value/2, f"{value:.2f}", ha='center', va='bottom', fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
            else:
                ax.text(x[j], bottom_neg[j] + value/2, f"{value:.2f}", ha='center', va='top', fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        bottom_pos += [val if mask else 0 for val, mask in zip(values, pos_mask)]
        bottom_neg += [val if mask else 0 for val, mask in zip(values, neg_mask)]

    # Set the x-axis labels for each reward bar
    if bar_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels)

    # Customize the plot
    ax.axhline(0, color='black', linewidth=0.8)  # Add a horizontal line at y=0
    ax.set_ylabel("Reward Value")
    if show_title:
        ax.set_title(title)

    # Create a separate legend
    legend_fig, legend_ax = plt.subplots(figsize=legend_size)
    legend_ax.legend(*ax.get_legend_handles_labels(), loc='center', ncol=1, frameon=False)
    legend_ax.axis('off')
    legend_fig.subplots_adjust(top=0.5)
    legend_fig.suptitle("Array Labels", y=0.85)
    # ax.legend()
    if save:
        legend_fig.savefig(os.path.join(save_path, title + "legend.png"), bbox_inches='tight', pad_inches=0.1)
        fig.savefig(os.path.join(save_path, title + '.png'), dpi=300, bbox_inches='tight')

    # Remove the legend from the main plot
    ax.get_legend()

    # Display the plot
    plt.tight_layout()
    plt.show()



def plot_trajectory_and_distribution(data_sets, interval_width, labels, colors, title="Agent Analysis",
                                     save=False, save_path='/Users/Hunter/Development/Academic/UML/RL/Hasenfus-RL/Multi-Agent/maddpg/experiments/plots', show_title=True, xlim=None, ylim=None):
    """
    Plot trajectories and distance distributions for multiple datasets in separate subplots.

    :param data_sets: A list of tuples, where each tuple contains (trajectories, distances) for a dataset.
    :param interval_width: The width of each interval for the distance distribution plot.
    :param labels: A list of labels for each dataset.
    :param colors: A list of colors for each dataset.
    :param title: The title of the plot.
    :param save: Whether to save the plot and legend as image files.
    :param save_path: The path to save the plot and legend images.
    :param show_title: Whether to display the title on the plot.
    """

    num_datasets = len(data_sets)
    # fig, axes = plt.subplots(num_datasets, 2, figsize=(8, num_datasets * 4))
    fig, axes = plt.subplots(2, num_datasets,figsize=(4 * num_datasets, 6))


    # Setting the plot's rcParams
    plt.rcParams.update({
        'font.size': 12,
        'lines.linewidth': 2,
        'axes.labelsize': 12,  # Axis label size
        'axes.titlesize': 14,  # Title size
        'figure.autolayout': True,  # Enable automatic layout adjustment
    })

    # Plotting trajectories and distance distributions for each dataset
    for i, (trajectories, distances) in enumerate(data_sets):
        color = colors[i]
        label = labels[i]

        # Plotting trajectories
        ax1 = axes[0,i]
        for traj in trajectories:
            x_coords, y_coords = zip(*traj)
            ax1.plot(x_coords, y_coords, color=color, label=label)
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        # ax1.set_title(f"Trajectory - {label}")
        ax1.grid(True)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        if xlim:
            ax1.set_xlim(xlim)
        if ylim:
            ax1.set_ylim(ylim)

        # Plotting distance distribution
        ax2 = axes[1, i]
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        bins = np.arange(min_distance, max_distance + interval_width, interval_width)
        counts, _ = np.histogram(distances, bins=bins)
        percentages = (counts / counts.sum()) * 100
        bar_labels = [f'{int(bins[i])}-{int(bins[i + 1])}' for i in range(len(bins) - 1)]
        ax2.bar(bar_labels, percentages, width=0.8, color=color, edgecolor='black', label=label)
        ax2.set_xlabel("Distance Intervals")
        ax2.set_ylabel("Percentage of Runs (%)")
        # ax2.set_title(f"Distance Distribution - {label}")
        ax2.set_xticklabels(bar_labels, rotation=45)
        ax2.grid(axis='y', linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

    # Save the plot if 'save' is True
    if save:
        fig.savefig(os.path.join(save_path, title + '.png'), dpi=300, bbox_inches='tight')


    # Creating and saving legend separately
    legend_fig, legend_ax = plt.subplots(figsize=(3, len(labels) * 0.3))
    legend_handles = [plt.Line2D([0], [0], color=color, lw=2) for color in colors]
    legend_ax.legend(legend_handles, labels, loc='center left', frameon=False)
    legend_ax.axis('off')
    if save:
        legend_fig.savefig(os.path.join(save_path, title + '_legend.png'), dpi=300, bbox_inches='tight')

    # Show the plot
    plt.tight_layout()
    if show_title:
        fig.suptitle(title)
    plt.show()


import matplotlib.pyplot as plt

def create_legend(colors, labels, save=False, save_path='/Users/Hunter/Development/Academic/UML/RL/Hasenfus-RL/Multi-Agent/maddpg/experiments/plots', title="Legend"):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(1, 1))

    # Create a dummy plot for each color and label
    for color, label in zip(colors, labels):
        ax.plot([], [], color=color, label=label)

    # Hide the axis
    ax.axis('off')

    # Create the legend
    legend = ax.legend(loc='center', frameon=False)

    # Adjust the legend font size if needed
    for text in legend.get_texts():
        text.set_fontsize('large')

    # Save the legend if 'save' is True
    if save:
        fig.savefig(os.path.join(save_path, title + '.png'))

    plt.show()



def plot_combined_analysis(mean_values_list, confidence_intervals_list, timesteps, data_sets, interval_width, labels, colors, title="Agent Analysis",xlabel='Episodes',ylabel='Reward',
                           save=False, save_path='/Users/Hunter/Development/Academic/UML/RL/Hasenfus-RL/Multi-Agent/maddpg/experiments/plots',
                           show_title=True, xlim=None, ylim=None, smoothing_factor=0.0, graph_ylim=None):
    """
    Plot graphs, trajectories, and distance distributions for multiple datasets in a descending order.

    :param mean_values_list: List of arrays of mean values for each algorithm.
    :param confidence_intervals_list: List of arrays of confidence intervals for each algorithm.
    :param timesteps: Array of timesteps.
    :param data_sets: A list of tuples, where each tuple contains (trajectories, distances) for a dataset.
    :param interval_width: The width of each interval for the distance distribution plot.
    :param labels: A list of labels for each dataset.
    :param colors: A list of colors for each dataset.
    :param title: The title of the plot.
    :param save: Whether to save the plot and legend as image files.
    :param save_path: The path to save the plot and legend images.
    :param show_title: Whether to display the title on the plot.
    :param xlim: The x-axis limits for the trajectory plot.
    :param ylim: The y-axis limits for the trajectory plot.
    :param smoothing_factor: Smoothing factor between 0 and 1.
    """
    num_datasets = len(data_sets)
    fig, axes = plt.subplots(num_datasets, 3, figsize=(12, num_datasets * 4))

    # Setting the plot's rcParams
    plt.rcParams.update({
        'font.size': 12,
        'lines.linewidth': 2,
        'axes.labelsize': 12,  # Axis label size
        'axes.titlesize': 14,  # Title size
        'figure.autolayout': True,  # Enable automatic layout adjustment
    })

    for i, (mean_values, confidence_interval, (trajectories, distances), label) in enumerate(zip(mean_values_list, confidence_intervals_list, data_sets, labels)):
        color = colors[i]
        print(mean_values.shape, confidence_interval.shape, len(trajectories), len(distances))
        # Plotting graph
        ax1 = axes[i, 0]
        smoothed_means = smooth_data(mean_values, smoothing_factor)
        smoothed_ci = smooth_data(confidence_interval, smoothing_factor)
        upper_bound = smoothed_means + smoothed_ci
        lower_bound = smoothed_means - smoothed_ci
        print(timesteps[-len(smoothed_means):].shape, smoothed_means.shape)
        ax1.plot(timesteps[-len(smoothed_means):], smoothed_means, label=f"Mean - {label}", color=color)
        ax1.fill_between(timesteps[-len(smoothed_means):], lower_bound, upper_bound, alpha=0.2, label=f"95% CI - {label}", color=color)
        if graph_ylim:
            ax1.set_ylim(graph_ylim)
        if xlabel is not None:
            ax1.set_xlabel(xlabel)
        else:
            ax1.set_xlabel("Timestep")
        if ylabel is not None:
            ax1.set_ylabel(ylabel)
        else:
            ax1.set_ylabel("Value")
        ax1.grid(True)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Plotting trajectories
        ax2 = axes[i, 1]
        for traj in trajectories:
            x_coords, y_coords = zip(*traj)
            ax2.plot(x_coords, y_coords, color=color, label=label)
        ax2.set_xlabel("X Position")
        ax2.set_ylabel("Y Position")
        ax2.grid(True)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        if xlim:
            ax2.set_xlim(xlim)
        if ylim:
            ax2.set_ylim(ylim)

        # Plotting distance distribution
        ax3 = axes[i, 2]
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        bins = np.arange(min_distance, max_distance + interval_width, interval_width)
        counts, _ = np.histogram(distances, bins=bins)
        percentages = (counts / counts.sum()) * 100
        bar_labels = [f'{int(bins[i])}-{int(bins[i + 1])}' for i in range(len(bins) - 1)]
        ax3.bar(bar_labels, percentages, width=0.8, color=color, edgecolor='black', label=label)
        ax3.set_xlabel("Distance Intervals")
        ax3.set_ylabel("Percentage of Runs (%)")
        ax3.set_xticklabels(bar_labels, rotation=45)
        ax3.grid(axis='y', linestyle='--')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

    # Save the plot if 'save' is True
    if save:
        fig.savefig(os.path.join(save_path, title + '.png'), dpi=300, bbox_inches='tight')

    # Creating and saving legend separately
    legend_fig, legend_ax = plt.subplots(figsize=(3, len(labels) * 0.3))
    legend_handles = [plt.Line2D([0], [0], color=color, lw=2) for color in colors]
    legend_ax.legend(legend_handles, labels, loc='center left', frameon=False)
    legend_ax.axis('off')
    if save:
        legend_fig.savefig(os.path.join(save_path, title + '_legend.png'), dpi=300, bbox_inches='tight')

    # Show the plot
    plt.tight_layout()
    if show_title:
        fig.suptitle(title)
    plt.show()