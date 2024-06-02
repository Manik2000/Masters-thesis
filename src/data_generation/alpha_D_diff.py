import argparse
import os
import pickle

import numpy as np
from andi_datasets.utils_challenge import label_continuous_to_list
from tqdm import tqdm

from src.datasets import AndiDataset


def set_up_parser():
    parser = argparse.ArgumentParser(description="Threshold tuning for change point detection")
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

    alphas = np.arange(1.8, 0.8, -0.1)
    diffs = np.arange(0.1, 1.1, 0.1)

    alpha_dict = {}

    for alpha_max, diff in tqdm(zip(alphas, diffs), total=len(alphas), desc="alpha"):
        alpha_dict[diff] = {}
        alpha_dict[diff]["traj"] = []
        alpha_dict[diff]["cp"] = []
        for _ in range(N):
            alpha = np.random.uniform(0.001, alpha_max)
            traj, labels = generator.multi_state(T=T, N=1, alphas=[[alpha, 0], [alpha+diff, 0]], Ds=[[1, 0], [1, 0]])
            alpha_dict[diff]["traj"].append(traj)
            alpha_dict[diff]["cp"].append(label_continuous_to_list(labels[:, 0, :])[0])

    with open(os.path.join(start, "alpha.pkl"), "wb") as file:
        pickle.dump(alpha_dict, file)

    Ds = np.logspace(-5, 2, 10)
    multiplies = [2, 5, 10, 25, 50, 100, 250, 500, 1000]

    Ds_dict = {}

    for multiply in tqdm(multiplies, desc="D"):
        Ds_dict[multiply] = {}
        Ds_dict[multiply]["traj"] = []
        Ds_dict[multiply]["cp"] = []

        for _ in range(N):
            D = np.random.choice(Ds) * np.random.uniform(0, 10)
            D2 = D * multiply
            traj, labels = generator.multi_state(T=T, N=1, alphas=[[1.5, 0], [1.5, 0]], Ds=[[D, 0], [D2, 0]])
            Ds_dict[multiply]["traj"].append(traj)
            Ds_dict[multiply]["cp"].append(label_continuous_to_list(labels[:, 0, :])[0])

    with open(os.path.join(start, "Ds.pkl"), "wb") as file:
        pickle.dump(Ds_dict, file)