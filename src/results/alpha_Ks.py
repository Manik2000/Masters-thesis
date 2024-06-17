import json
import os
import pickle

import numpy as np
from tqdm import tqdm

from src.cp_detector import CPDetector
from src.metrics import f1_score, rmse


if __name__ == "__main__":
    cp_detector = CPDetector(os.path.join("models", "cnn_lstm_attention_15.keras"), 15, threshold=0.9)
    start_path = os.path.join("data", "alpha_Ks_diffs")

    with open(os.path.join(start_path, "alpha.pkl"), "rb") as file:
        alphas_dict = pickle.load(file)
    with open(os.path.join(start_path, "Ks.pkl"), "rb") as file:
        Ks_dict = pickle.load(file)

    alphas_res = {}  

    data_dicts = [alphas_dict, Ks_dict]
    results = [{}, {}]

    for result_dict, data_dict, name in tqdm(
        zip(results, data_dicts, ["alpha", "Ks"]), total=2, desc="alpha or K"
    ):
        for key in tqdm(data_dict.keys()):
            result_dict[key] = {}
            content = data_dict[key]
            true_cp = content["cp"]
            trajs = content["traj"]

            preds = [np.array(cp_detector(trajs[:, i, :])) for i in range(trajs.shape[1])]

            rmse_ = np.mean(
                [rmse(np.array(true), pred) for (true, pred) in zip(true_cp, preds)]
            )
            f1_score_ = np.mean(
                [f1_score(np.array(true), pred) for (true, pred) in zip(true_cp, preds)]
            )

            result_dict[key]["rmse"] = rmse_
            result_dict[key]["f1_score"] = f1_score_
        with open(os.path.join("data", "results", f"{name}.json"), "w") as file:
            json.dump(result_dict, file)
