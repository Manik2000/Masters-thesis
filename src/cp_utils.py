import numpy as np
from andi_datasets.utils_challenge import label_continuous_to_list


def assign_to_groups(changepoints, window_length):
    """
    Assign changepoints to groups so that the total inertia is minimized 
    and the distance between the first and last point inside the group is at most window_length.
    Use greedy algorithm to find the best assignment.
    """
    labels = np.array([i // 15 for i in changepoints]).astype(int)
    n = len(changepoints)
    current_cost = 0
    for i in np.unique(labels):
        group = changepoints[labels == i]
        current_cost += np.sum((group - group.mean()) ** 2)
    current_best = current_cost
    while True:
        best_labels = labels.copy()
        for i in range(n):
            changepoint_i = changepoints[i]
            for j in range(i+1, n):
                changepoint_j = changepoints[j]
                if abs(changepoint_i - changepoint_j) < window_length:
                    if labels[i] != labels[j]:
                        new_labels = labels.copy()
                        new_labels[i] = labels[j]
                        new_cost = 0
                        for k in np.unique(new_labels):
                            group = changepoints[new_labels == k]
                            group_mean = np.mean(group)
                            new_cost += np.sum((group - group_mean) ** 2)
                        if new_cost < current_best:
                            current_best = new_cost
                            best_labels = new_labels
                else:
                    continue
        if current_best < current_cost:
            labels = best_labels
            current_cost = current_best
        else:
            break
    return labels


def merge_changepoints(changepoints, window_length):
    labels = assign_to_groups(changepoints, window_length)
    return np.array([round(np.mean(changepoints[labels == i])) for i in np.unique(labels)])


def get_changepoints(trajectory_labels, window_length):
    changepoints = np.array(label_continuous_to_list(trajectory_labels)[0]) - 1
    return merge_changepoints(changepoints, window_length)


def create_window_dataset(trajectory, changepoints, window_length):
    X = []
    y = []
    for i in range(len(trajectory) - window_length):
        X.append(trajectory[i:i+window_length, :])
        y.append(1 if any(cp in range(i, i+window_length) for cp in changepoints) else 0)
    return np.array(X), np.array(y)


def create_dataset(trajs, labels, window_length):
    X = []
    y = []
    changepoints = [get_changepoints(labels[:, i, :], window_length=window_length) for i in range(trajs.shape[1])]
    for i in range(trajs.shape[1]):
        Xi, yi = create_window_dataset(trajs[:, i, :], changepoints[i], window_length)
        X.append(Xi)
        y.append(yi)
    return np.concatenate(X), np.concatenate(y)
