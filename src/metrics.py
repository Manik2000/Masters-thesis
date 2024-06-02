import numpy as np
from scipy.optimize import linear_sum_assignment


def gated_distance(a, b, eps=10):
    return np.minimum(np.abs(a - b), eps)


def cost_matrix(cp_gt, cp_p, eps=10):
    """
    Return a cost matrix for predicted change points and ground truth ones.
    """
    cm = gated_distance(cp_gt.reshape((-1, 1)), cp_p.reshape((1, -1)), eps=eps)
    return cm


def minimal_paired_distances(cp_gt, cp_p, eps=10):
    cm = cost_matrix(cp_gt, cp_p, eps=eps)
    row_ind, col_ind = linear_sum_assignment(cm)
    return cm[row_ind, col_ind].sum()


def alpha_cp(cp_gt, cp_p, eps=10):
    """
    Return a value of a metric checking the distance between correctly paired points
    More in https://arxiv.org/pdf/2311.18100
    """
    d_cp = minimal_paired_distances(cp_gt, cp_p)
    return 1 - d_cp / (len(cp_gt) * eps)


def rmse(cp_gt, cp_p, eps=10):
    """
    Calculate RMSE for correctly paired ground truth and predicted change points.
    More in https://arxiv.org/pdf/2311.18100
    """
    cm = cost_matrix(cp_gt, cp_p)
    row_ind, col_ind = linear_sum_assignment(cm)
    costs = cm[row_ind, col_ind] < eps
    paired_gt = cp_gt[row_ind][costs]
    paired_p = cp_p[col_ind][costs]
    return np.sqrt(np.mean((paired_gt - paired_p) ** 2))


def jaccard(cp_gt, cp_p, eps=10):
    """
    Calculate the jaccard score for change points predictions.
    More in https://arxiv.org/pdf/2311.18100
    """
    cm = cost_matrix(cp_gt, cp_p, eps=eps)
    row_ind, col_ind = linear_sum_assignment(cm)
    tp = sum(cm[i, j] < eps for (i, j) in zip(row_ind, col_ind))
    fp = len(cp_p) - tp
    fn = len(cp_gt) - tp
    return tp / (tp + fp + fn)


def annotation_error(cp_gt, cp_p):
    """
    Calculate the annotation error for change points predictions.
    """
    return abs(len(cp_gt) - len(cp_p))


def f1_score(cp_gt, cp_p, eps=10):
    """
    Calculate the F1 score for change points predictions.
    """
    cm = cost_matrix(cp_gt, cp_p, eps=eps)
    row_ind, col_ind = linear_sum_assignment(cm)
    tp = sum(cm[i, j] < eps for (i, j) in zip(row_ind, col_ind))
    fp = len(cp_p) - tp
    fn = len(cp_gt) - tp
    return 2 * tp / (2 * tp + fp + fn)


if __name__ == "__main__":
    cp_p = np.array([-5, 20, 33, 60])
    cp_gt = np.array([6, 32, 60])
    print(rmse(cp_gt, cp_p))
    print(alpha_cp(cp_gt, cp_p))
    print(jaccard(cp_gt, cp_p))
    print(f1_score(cp_gt, cp_p))
    print(annotation_error(cp_gt, cp_p))
