import argparse
import json
import logging
import os

import numpy as np
import tensorflow as tf

from src.models.all_models import (
    generate_cnn,
    generate_cnn_lstm,
    generate_cnn_lstm_attention,
    generate_lstm,
)

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    all_models = {
        "cnn": generate_cnn,
        "lstm": generate_lstm,
        "cnn_lstm": generate_cnn_lstm,
        "cnn_lstm_attention": generate_cnn_lstm_attention,
    }

    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("-model", type=str, default="cnn", help="Model to train")
    parser.add_argument(
        "-window_length",
        type=int,
        default=15,
        help="Length of the window for the classifier",
    )
    parser.add_argument("-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    window_length = args.window_length

    models = list(map(lambda x: x.strip(), args.model.split(",")))
    print(models)
    if models == ["all"]:
        models = list(all_models.keys())

    path = os.path.join("data", "classifier", f"{window_length}")
    train_X = np.load(os.path.join(path, "train", "X_train.npy"))
    train_y = np.load(os.path.join(path, "train", "y_train.npy"))
    val_X = np.load(os.path.join(path, "val", "X_val.npy"))
    val_y = np.load(os.path.join(path, "val", "y_val.npy"))

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_X, train_y))
        .batch(args.batch_size)
        .shuffle(len(train_X))
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices((val_X, val_y))
        .batch(args.batch_size)
        .shuffle(len(val_X))
    )

    logger.info(f"Training models: {', '.join(models)}")
    logger.info(f"Training with window: {window_length}")

    for model in models:
        logger.info(f"Training model: {model}")
        dl_model = all_models[model](args.window_length)
        dl_hist = dl_model.fit(
            train_dataset, epochs=args.epochs, validation_data=val_dataset
        )
        with open(
            os.path.join(
                "data", "results", f"{model}_{window_length}_training_results.json"
            ),
            "w",
        ) as f:
            json.dump(dl_hist.history, f)
        dl_model.save(f"models/{model}_{window_length}.keras")
        tf.keras.backend.clear_session()
