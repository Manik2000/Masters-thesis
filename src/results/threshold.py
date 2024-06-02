import argparse
import json
import os
import pickle

import numpy as np
from tqdm import tqdm

from src.cp_detector import CPDetector
from src.metrics import f1_score, rmse


def set_up_parser():
    parser = argparse.ArgumentParser(
        description="Threshold tuning for change point detection"
    )
    parser.add_argument(
        "-length", type=int, default=85, help="Length of the trajectory"
    )


if __name__ == "__main__":
    thresholds = np.arange(0.4, 1, 0.05)

    parser = set_up_parser()
    args = parser.parse_args()
    length = args.length

    data_path = os.path.join("..", "data", "final_eval", f"{length}", "val")
    data = np.load(os.path.join(data_path, "X.npy"))
    with open(os.path.join(data_path, "change_points.pkl"), "rb") as f:
        change_points = pickle.load(f)

    window_lengths = [10, 15, 20]
    model_paths = [
        os.path.join("models", f"cnn_lstm_attention_{window_length}.keras")
        for window_length in window_lengths
    ]
    END = [length]

    all_results = {}

    for window_length, model_path in tqdm(
        zip(window_lengths, model_paths),
        desc="Evaluating models",
        total=len(window_lengths),
    ):
        cp_det = CPDetector(model_path, window_length)
        prob_seq = [cp_det.get_probabilities(i) for i in data]
        f1_scores = []
        rmse_scores = []
        for threshold in tqdm(thresholds, desc="Threshold"):
            cp_det.threshold = threshold
            preds = [
                cp_det.get_change_points_preds(prob_seq_i) + END
                for prob_seq_i in prob_seq
            ]
            f1_scores.append(
                np.mean(
                    [
                        f1_score(np.array(change_points[i]), np.array(preds[i]))
                        for i in range(len(preds))
                    ]
                )
            )
            rmse_scores.append(
                np.mean(
                    [
                        rmse(np.array(change_points[i]), np.array(preds[i]))
                        for i in range(len(preds))
                    ]
                )
            )
        all_results[window_length] = {"f1": f1_scores, "rmse": rmse_scores}

    with open(os.path.join("..", "data", "results", "thresholds.json"), "w") as f:
        json.dump(all_results, f)
