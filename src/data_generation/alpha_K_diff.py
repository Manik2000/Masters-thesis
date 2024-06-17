import argparse
import os
import pickle

import numpy as np
from andi_datasets.utils_challenge import label_continuous_to_list
from tqdm import tqdm

from src.datasets import AndiDataset


def set_up_parser():
    parser = argparse.ArgumentParser(
        description="Threshold tuning for change point detection"
    )
    parser.add_argument("-T", type=int, default=90, help="Length of the trajectory")
    parser.add_argument("-N", type=int, default=100, help="Number of trajectories")
    return parser


if __name__ == "__main__":
    start = os.path.join("data", "alpha_Ds_diffs")
    os.makedirs(start, exist_ok=True)

    parser = set_up_parser()
    args = parser.parse_args()
    T = args.T
    N = args.N

    generator = AndiDataset()
    alpha0 = 0.3

    diffs = np.arange(0.1, 1.6, 0.1)

    alpha_dict = {}

    for diff in tqdm(diffs, total=len(diffs), desc="alpha"):
        alpha_dict[diff] = {}
        trajs, labels = generator.multi_state(
            T=T, N=N, alphas=[[alpha0, 0], [alpha0 + diff, 0]], Ds=[[1, 0], [1, 0]]
        )
        alpha_dict[diff]["traj"] = trajs
        alpha_dict[diff]["cp"] = [label_continuous_to_list(labels[:, i, :])[0] for i in range(N)]

    with open(os.path.join(start, "alpha.pkl"), "wb") as file:
        pickle.dump(alpha_dict, file)

    K0 = 1e-3
    multiplies = [2, 5, 10, 25, 50, 75, 100, 200, 250, 500, 750, 1000]

    Ks_dict = {}

    for multiply in tqdm(multiplies, desc="K"):
        Ks_dict[multiply] = {}

        trajs, labels = generator.multi_state(
            T=T, N=N, alphas=[[1.2, 0], [1.2, 0]], Ds=[[K0, 0], [K0 * multiply, 0]]
        )
        Ks_dict[multiply]["traj"] = trajs
        Ks_dict[multiply]["cp"] = [label_continuous_to_list(labels[:, i, :])[0] for i in range(N)]

    with open(os.path.join(start, "Ks.pkl"), "wb") as file:
        pickle.dump(Ks_dict, file)
