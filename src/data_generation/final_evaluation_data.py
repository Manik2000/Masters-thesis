import os
import pickle

import numpy as np
from rich.progress import Progress
from sklearn.model_selection import train_test_split

from src.data_generation.classifier_data import generate_trajectory_from_model
from src.datasets import AndiDataset

M = 800
lengths = [50, 85, 120]
model_names = [
    "single_state",
    "multi_state",
    "confinemnet",
    "immobile",
    "dimmerization",
]
thinning = [False, False, False, False, True]
Ns = [2, 2, 2, 2, 150]


def generate_changepoints_data(
    generator: AndiDataset,
    model_name: str,
    traj_lentgh: int,
    N: int,
    thinning: bool = False,
    M=M,
):
    model = getattr(generator, model_name)
    model_X = []
    model_change_points = []
    with Progress() as progress:
        task = progress.add_task(f"{model_name}", total=M)
        while not progress.finished:
            is_valid = False
            while not is_valid:
                try:
                    if model_name == "immobile":
                        alphas, Ds = [
                            generator.random_alpha_value(),
                            np.random.uniform(0, 0.15),
                        ], [generator.random_D_value(), np.random.uniform(0, 0.15)]
                    elif model_name == "single_state":
                        alphas, Ds = (
                            generator.random_alpha_value(),
                            generator.random_D_value(),
                        )
                    else:
                        alphas, Ds = generator.get_alphas_and_Ds()
                    trajs, labels = generate_trajectory_from_model(
                        model, N=N, T=traj_lentgh, alphas=alphas, Ds=Ds
                    )
                    is_valid = True
                except ValueError:
                    continue
            if thinning:
                indices = np.random.choice(trajs.shape[1], size=2, replace=False)
                trajs = trajs[:, indices, :]
                labels = labels[:, indices, :]
            model_X.append(trajs)
            changepoints = generator.get_changepoints(labels)
            model_change_points.extend(changepoints)
            n = len(changepoints)
            progress.update(task, advance=n)
    return np.concatenate(model_X, axis=1), model_change_points


def get_changepoints_per_traj_length(
    dataset_generator: AndiDataset, length: int, M: int = M
):
    all_X = []
    all_change_points = []
    all_labels = []
    for i, model_name in enumerate(model_names):
        X, change_points = generate_changepoints_data(
            dataset_generator, model_name, length, Ns[i], thinning[i], M=M
        )
        all_X.append(X)
        all_change_points.extend(change_points)
        all_labels.extend(model_name for _ in range(len(change_points)))
    all_X = np.concatenate(all_X, axis=1).transpose(1, 0, 2)

    val_X, test_X, val_change_points, test_change_points, val_labels, test_labels = (
        train_test_split(all_X, all_change_points, all_labels, test_size=0.8)
    )

    path = os.path.join("data", "final_eval", f"{length}")
    for i in ["val", "test"]:
        os.makedirs(os.path.join(path, i), exist_ok=True)

    np.save(os.path.join(path, "val", "X.npy"), val_X)
    np.save(os.path.join(path, "test", "X.npy"), test_X)
    with open(os.path.join(path, "val", "change_points.pkl"), "wb") as f:
        pickle.dump(val_change_points, f)
    with open(os.path.join(path, "test", "change_points.pkl"), "wb") as f:
        pickle.dump(test_change_points, f)
    with open(os.path.join(path, "val", "labels.pkl"), "wb") as f:
        pickle.dump(val_labels, f)
    with open(os.path.join(path, "test", "labels.pkl"), "wb") as f:
        pickle.dump(test_labels, f)


def main():
    dataset_generator = AndiDataset()
    for length in lengths:
        get_changepoints_per_traj_length(dataset_generator, length)


if __name__ == "__main__":
    main()
