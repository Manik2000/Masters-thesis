import numpy as np


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