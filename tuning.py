import os
import random
import tempfile
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import ray
import ray.tune as tune
import torch
from hyperopt import hp
from ray.tune import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from dataset import EmbeddingCollate, IronyDetectionDataset
from model import IronyClassifier
from training import eval, train


def objective(
    config: dict,
    emb_model: str = "bert-cls",
):

    DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_SPLIT = 0.8

    # Loading data from csv
    base_dir = os.path.abspath(os.path.dirname(__file__))
    train_data_path = os.path.join(base_dir, config["PATH"])
    data_df = pd.read_csv(train_data_path)

    # Instantiating dataset
    full_dataset = IronyDetectionDataset(
        data=data_df,
    )

    # Getting splits
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    workers = 0 if torch.cuda.is_available() else 8

    collate_func = EmbeddingCollate(
        embedding_model=emb_model, multiclass=config["MULTI"], device=DEVICE_NAME
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        num_workers=workers,
        collate_fn=collate_func,  # pyright:ignore
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["BATCH_SIZE"],
        num_workers=workers,
        collate_fn=collate_func,  # pyright:ignore
    )

    input_shape = next(iter(train_dataloader))[0].size(
        -1
    )  # getting the embeddings size to determine the model's input dimension

    # Initializing model
    print("Model input dim: ", input_shape)
    model = IronyClassifier(
        emb_dim=input_shape,
        h_dim=config["H_DIM"],
        use_learnable_dr=True if config["REDUX"] else False,
        reduced_dim=config["REDUX"],
        n_classes=1 if not config["MULTI"] else 4,
        drop_rate=config["DROP_RATE"],
    )
    model.to(DEVICE_NAME)

    # Getting training components
    scaler = torch.GradScaler("cuda", enabled=(DEVICE_NAME == "cuda"))

    if not config["MULTI"]:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:

        train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
        label_series = pd.Series(train_labels)
        label_counts = label_series.value_counts().sort_index().to_numpy()

        weights = 1.0 / label_counts  # inverse frequency
        weights = weights / weights.sum()  # Normalize
        weights_tensor = torch.FloatTensor(weights).to(DEVICE_NAME)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)

    if config["OPT"] == "SGD":
        optim = torch.optim.SGD(
            params=model.parameters(),
            lr=config["LR"],
            weight_decay=config["WEIGHT_DECAY"],
        )
    else:
        optim = torch.optim.AdamW(
            params=model.parameters(),
            lr=config["LR"],
            weight_decay=config["WEIGHT_DECAY"],
        )

    plat_scheduler = ReduceLROnPlateau(
        optimizer=optim,
        mode="max",
        patience=config["PATIENCE"],
    )

    loaded_checkpoint = get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            checkpoint_data = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(checkpoint_data["model_state"])
            optim.load_state_dict(checkpoint_data["optimizer_state"])
            scaler.load_state_dict(checkpoint_data["scaler_state"])
    best_f1 = -1.0
    for epoch in range(config["MAX_EPOCHS"]):
        _ = train(
            multiclass=config["MULTI"],
            dataloader=train_dataloader,
            model=model,
            criterion=loss_fn,
            optimizer=optim,
            device=DEVICE_NAME,
            scaler=scaler,
            l1_coeff=config["L1"],
        )

        val_result = eval(
            multiclass=config["MULTI"],
            val_dataloader=val_dataloader,
            model=model,
            criterion=loss_fn,
            device=DEVICE_NAME,
        )

        plat_scheduler.step(val_result["f1"])
        if val_result["f1"] > best_f1:
            best_f1 = val_result["f1"]
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint.pt")
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optim.state_dict(),
                        "scaler_state": scaler.state_dict(),
                    },
                    path,
                )
                checkpoint = Checkpoint.from_directory(checkpoint_dir)

                tune.report(
                    {
                        "l": val_result["loss"],
                        "f1": val_result["f1"],
                    },
                    checkpoint=checkpoint,
                )
        else:
            tune.report(
                {
                    "l": val_result["loss"],
                    "f1": val_result["f1"],
                },
            )

    print("Finished!")


def tune_downstream_model(
    config, embedding_model, trainable=objective, bayesian=True, num_samples=10
):

    timestamp = datetime.now().strftime("%H_%M")

    static_params = config["STATIC"]
    scheduler = ASHAScheduler(
        max_t=static_params["MAX_EPOCHS"],
        grace_period=5,
        reduction_factor=2,
        metric="f1",
        mode="max",
    )
    if bayesian:
        searcher = HyperOptSearch(space=config["HYPEROPT"], metric="f1", mode="max")
        params = {**static_params}
    else:
        searcher = None
        params = {**config["TUNE"], **static_params}

    # Remove the environment variable
    if "AIR_VERBOSITY" in os.environ:
        print("Reset air verbosity")
        del os.environ["AIR_VERBOSITY"]

    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    tuner = tune.Tuner(
        tune.with_resources(
            partial(trainable, emb_model=embedding_model),
            resources={"cpu": 8, "gpu": 1},
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=searcher,
            num_samples=num_samples,
            trial_name_creator=lambda trial: trial.trial_id[-4:],
            trial_dirname_creator=lambda trial: trial.trial_id,
        ),
        param_space=params,
        run_config=tune.RunConfig(
            name=f"{embedding_model}_exp_{timestamp}",
            progress_reporter=tune.CLIReporter(
                metric_columns=["l", "f1"],
                max_column_length=5,
                metric="f1",
                mode="max",
                sort_by_metric=True,
            ),
        ),
    )

    # Setting seeds for reproducibility
    SEED = static_params.get("SEED", 42)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Running tuning optimization
    result = tuner.fit()

    # Getting top-5 trials metrics&config
    os.makedirs("results", exist_ok=True)
    result_df = result.get_dataframe(filter_metric="f1", filter_mode="max").sort_values(
        by=["f1"]
    )
    result_df.head().to_csv(
        f"./results/trials_df_{embedding_model}_{timestamp}.csv"  # saving as csv for plotting or further analysis
    )  # storing as csv
    best_trial = result.get_best_result("f1", "max")

    if best_trial.config is None or best_trial.metrics is None:
        raise AttributeError(
            "Best trial has unexpected 'None' attributes (.config, .metrics or .checkpoint). Print them for debugging"
        )

    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.metrics['l']}")
    print(f"Best trial final validation f1-score: {best_trial.metrics['f1']}")

    return best_trial


if __name__ == "__main__":

    MODEL = "instructor"

    config = {
        "STATIC": {
            "PATH": "./data/train/SemEval2018-T3-train-taskB_emoji.csv",
            "MULTI": True,  # wether to perform binary or multiclass classification
            "OPT": "AdamW",  # tune.choice(["SGD", "ADAM"]),
            "MAX_EPOCHS": 20,
            "SAMPLES": 10,
            "SEED": 42,
        },
        "HYPEROPT": {
            "REDUX": hp.choice("REDUX", [None, 128, 256, 512]),
            "H_DIM": hp.choice("H_DIM", [8, 16, 32, 64, 128]),
            "LR": hp.loguniform("LR", np.log(2e-5), np.log(5e-3)),
            "WEIGHT_DECAY": hp.loguniform("WEIGHT_DECAY", np.log(1e-5), np.log(5e-1)),
            "L1": hp.loguniform("L1", np.log(1e-6), np.log(5e-4)),
            "DROP_RATE": hp.choice("DROP_RATE", [0.2, 0.3, 0.4, 0.5]),
            "BATCH_SIZE": hp.choice("BATCH_SIZE", [8, 16, 32, 64]),
            "PATIENCE": hp.randint("PATIENCE", 4) + 2,  # rand int in [2, 6)
        },
        # HYPERPARAMS
        "TUNE": {
            "REDUX": tune.choice([None, 128, 256, 512]),
            "H_DIM": tune.choice([2**i for i in range(3, 8)]),
            "LR": tune.loguniform(2e-5, 5e-3),
            "WEIGHT_DECAY": tune.loguniform(1e-5, 5e-1),
            "L1": tune.loguniform(1e-6, 5e-4),
            "DROP_RATE": tune.choice([0.2, 0.3, 0.4, 0.5]),
            "BATCH_SIZE": tune.choice([8, 16, 32, 64]),
            "PATIENCE": tune.randint(2, 6),
        },
    }

    tune_downstream_model(
        config, embedding_model=MODEL, num_samples=config["STATIC"]["SAMPLES"]
    )
