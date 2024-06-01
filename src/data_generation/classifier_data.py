import argparse
import os

import numpy as np
import stochastic
from rich.progress import Progress
from sklearn.model_selection import train_test_split

from src.cp_utils import create_dataset
from src.datasets import AndiDataset

T = 150
L = 1.5*128


def reduce_the_number_of_no_change(X, y):
    change_point_n = np.sum(y == 1)
    max_no_change = min(2 * change_point_n, len(y) - change_point_n)
    no_change = np.where(y == 0)[0]
    no_change = np.random.choice(no_change, max_no_change, replace=False)
    X = np.concatenate([X[y == 1], X[no_change]])
    y = np.concatenate([y[y == 1], y[no_change]])
    return X, y


def reduce_the_number_of_change(X, y):
    no_change_point_n = np.sum(y == 0)
    max_change = min(no_change_point_n, len(y) - no_change_point_n)
    change = np.where(y == 1)[0]
    change = np.random.choice(change, max_change, replace=False)
    X = np.concatenate([X[y == 0], X[change]])
    y = np.concatenate([y[y == 0], y[change]])
    return X, y


def generate_trajectory_from_model(generator, N, T, alphas, Ds):
    return generator(N=N, T=T, alphas=alphas, Ds=Ds)


def generate_model_data(generator, model_name, window_length, N, thinning, change_points_per_category, no_change_points_per_category):
    X_model, y_model = [], []
    model = getattr(generator, model_name)
    with Progress() as progress:
        task1 = progress.add_task(f"{model_name} - Change", total=change_points_per_category)
        task2 = progress.add_task(f"{model_name} - No change", total=no_change_points_per_category)
        while not progress.finished:
            is_valid = False
            while not is_valid:
                try:
                    if model_name == "immobile":
                        alphas, Ds = [generator.random_alpha_value(), np.random.uniform(0, 0.15)], \
                                      [generator.random_D_value(), np.random.uniform(0, 0.15)]
                    else:
                        alphas, Ds = generator.get_alphas_and_Ds()
                    trajs, labels = generate_trajectory_from_model(model, N, T, 
                                                                   alphas=alphas, Ds=Ds)
                    is_valid = True
                except ValueError:
                    continue
            if thinning:
                indices = np.random.choice(trajs.shape[1], size=2, replace=False)
                trajs = trajs[:, indices, :]
                labels = labels[:, indices, :]
            X_new, y_new = create_dataset(trajs, labels, window_length)
            i = np.sum(y_new == 1)
            i2 = np.sum(y_new == 0)
            progress.update(task1, advance=i)
            progress.update(task2, advance=i2)
            X_model.extend(X_new)
            y_model.extend(y_new)
    return X_model, y_model


def set_up_parser():
    parser = argparse.ArgumentParser(description="Generate data for the classifier")
    parser.add_argument("-window_length", type=int, default=15, help="Length of the window for the classifier")
    return parser


def main(window_length):
    stochastic.random.seed(3)
    np.random.seed(7)

    trajs_per_category = 250_000
    change_points_per_category = int(trajs_per_category / 3)
    no_change_points_per_category = trajs_per_category - change_points_per_category

    generator = AndiDataset()
    X = []
    y = []
    labels = []

    all_model_names = ["multi_state", "confinemnet", "immobile", "dimmerization"]
    Ns = [2, 2, 2, 150]
    thinning = [False, False, False, True]

    for model_name, N, is_thinning in zip(all_model_names, Ns, thinning):
        X_model, y_model = generate_model_data(generator, model_name, window_length, N, is_thinning, change_points_per_category, no_change_points_per_category)
        X_model, y_model = np.stack(X_model), np.array(y_model)
        if np.mean(y_model) > 0.5:
            X_model, y_model = reduce_the_number_of_change(X_model, y_model)
        else:
            X_model, y_model = reduce_the_number_of_no_change(X_model, y_model)
        
        X.extend(X_model)
        y.extend(y_model)
        labels.extend(model_name for _ in range(len(y_model)))

    X = np.stack(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(X, y, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val, labels_train, labels_val = train_test_split(X_train, y_train, labels_train, test_size=0.2, random_state=42)

    path = os.path.join("data", "classifier", f"{window_length}")
    for i in ["train", "val", "test"]:
        os.makedirs(os.path.join(path, i), exist_ok=True)

    np.save(os.path.join("data", "classifier", f"{window_length}", "train", "X_train.npy"), X_train)
    np.save(os.path.join("data", "classifier", f"{window_length}", "train", "y_train.npy"), y_train)
    np.save(os.path.join("data", "classifier", f"{window_length}", "train", "labels_train.npy"), labels_train)

    np.save(os.path.join("data", "classifier", f"{window_length}", "val", "X_val.npy"), X_val)
    np.save(os.path.join("data", "classifier", f"{window_length}", "val", "y_val.npy"), y_val)
    np.save(os.path.join("data", "classifier", f"{window_length}", "val", "labels_val.npy"), labels_val)

    np.save(os.path.join("data", "classifier", f"{window_length}", "test", "X_test.npy"), X_test)
    np.save(os.path.join("data", "classifier", f"{window_length}", "test", "y_test.npy"), y_test)
    np.save(os.path.join("data", "classifier", f"{window_length}", "test", "labels_test.npy"), labels_test)



if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    window_length = args.window_length

    main(window_length)
