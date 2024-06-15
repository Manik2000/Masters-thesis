import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.base_detector import BaseDetector
from src.cp_detector import CPDetector
from src.metrics import alpha_cp, annotation_error, f1_score, jaccard, rmse

if __name__ == "__main__":

    traj_lengths = [50, 120]
    path = os.path.join("data", "final_eval")

    base_detector = BaseDetector(window_length=20)
    detectors = [BaseDetector(window_length=20)] + [
        CPDetector(
            model_path=os.path.join("models", f"cnn_lstm_attention_{window}.keras"),
            window_length=window,
            threshold=0.9
        )
        for window in [10, 15, 20]
    ]
    detectors_names = ["Baseline", "CPDetector10", "CPDetector15", "CPDetector20"]

    all_results = []
    for traj_length in tqdm(traj_lengths, desc="Trajectory length"):
        data = np.load(os.path.join(path, f"{traj_length}", "test", "X.npy"))
        with open(
            os.path.join(path, f"{traj_length}", "test", "change_points.pkl"), "rb"
        ) as f:
            change_points = [np.array(i) for i in pickle.load(f)]
        with open(
            os.path.join(path, f"{traj_length}", "test", "labels.pkl"), "rb"
        ) as f:
            labels = pickle.load(f)
        for idx, traj in tqdm(
            enumerate(data), desc="Trajectories cp", total=data.shape[0]
        ):
            preds = [np.array(detector.predict(traj)) for detector in detectors]
            traj_changepoints = change_points[idx]
            for j, detector in enumerate(detectors):
                results = {
                    "traj_length": traj_length,
                    "label": labels[idx],
                    "detector": detectors_names[j],
                    "f1": f1_score(traj_changepoints, preds[j]),
                    "rmse": rmse(traj_changepoints, preds[j]),
                    "jaccard": jaccard(traj_changepoints, preds[j]),
                    "annotation_error": annotation_error(traj_changepoints, preds[j]),
                    "alpha_cp": alpha_cp(traj_changepoints, preds[j]),
                }
                all_results.append(results)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join("data", "results", "final_eval_results.csv"), index=False
    )
