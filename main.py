import gc
import json
import os
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M")

import numpy as np
import pandas as pd
import torch
from hyperopt import hp
from ray import tune
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import EmbeddingCollate, IronyDetectionDataset
from model import IronyClassifier
from training import run_session, run_test_data, train
from tuning import tune_downstream_model


def full_data_training(model, config, multiclass: bool, data_path):

    DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE_NAME)
    EMB_MODEL = config["EMB_MODEL"]
    EPOCHS = config["MAX_EPOCHS"]
    LR = config["LR"]
    BATCH_SIZE = config["BATCH_SIZE"]
    WEIGHT_DECAY = config["WEIGHT_DECAY"]
    L1 = config["L1"]
    PATIENCE = config["PATIENCE"]
    SEED = config["SEED"]

    # Creating dir to save best model and config

    task_indicator = "A" if not config["MULTI"] else "B"
    session_path = f"models/final_session{task_indicator}_{EMB_MODEL}_{TIMESTAMP}"
    os.makedirs(session_path, exist_ok=True)

    # Setting seed for reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Getting data
    data_df = pd.read_csv(data_path)
    full_dataset = IronyDetectionDataset(
        data=data_df,
    )
    # Printing out label distributions
    print(data_df["Label"].value_counts())

    # Setting up dataloaders
    workers = 0 if DEVICE_NAME == "cuda" else 8

    collate_func = EmbeddingCollate(
        embedding_model=EMB_MODEL, multiclass=multiclass, device=DEVICE_NAME
    )

    full_dataloader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        num_workers=workers,
        collate_fn=collate_func,  # pyright:ignore
        shuffle=True,
    )

    input_shape = next(iter(full_dataloader))[0].size(
        -1
    )  # getting the embeddings size to determine the model's input dimension

    # Initializing model
    print("Model input dim: ", input_shape)

    scaler = torch.GradScaler("cuda", enabled=(DEVICE_NAME == "cuda"))

    if not multiclass:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        label_counts = data_df["Label"].value_counts().to_numpy()
        weights = 1.0 / label_counts  # inverse frequency
        weights = weights / weights.sum()  # Normalize
        weights_tensor = torch.FloatTensor(weights).to(DEVICE_NAME)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)

    optim = torch.optim.AdamW(
        params=model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optim,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    plat_scheduler = ReduceLROnPlateau(
        optimizer=optim,
        mode="min",
        patience=PATIENCE,
        factor=0.1,
    )

    all_f1 = 0.0
    all_acc = 0.0
    last_epoch = 0
    try:
        for epoch in range(EPOCHS):

            print("\n", "-" * 40, "\n")
            print("#" * 10, f" EPOCH: {epoch+1} ", "#" * 10, "\n")

            train_result = train(
                multiclass=multiclass,
                dataloader=full_dataloader,
                model=model,
                criterion=loss_fn,
                optimizer=optim,
                device=DEVICE_NAME,
                scaler=scaler,
                l1_coeff=L1,
            )
            loss, acc = train_result["loss"], train_result["accuracy"]
            print("\n", "*" * 10, "TRAIN METRICS", "*" * 10)
            print()
            print(train_result["report"])
            print(
                f"Running Loss | {train_result["loss"]:.5f}\nTrain Accuracy | {train_result["accuracy"]:.2f}%"
            )
            all_f1 += train_result["f1"]
            all_acc += acc
            last_epoch = epoch + 1

            if epoch >= EPOCHS // 2:
                plat_scheduler.step(loss)
            else:
                scheduler.step()

    except KeyboardInterrupt:
        print("\nSession stopped early by SIGINT\n")

    # Saving relevant data for current session
    checkpoint = {
        "model": model,
        "avg_acc": all_acc / last_epoch,
        "avg_f1": all_f1 / last_epoch,
    }

    torch.save(
        checkpoint,
        os.path.join(session_path, "best_model.pt"),
    )

    return checkpoint


def kfold_cv(config, k=5, stratified=True):
    """
    Function to perform k-fold cross validation with stratification support.

    Args:
        config: Configuration dictionary containing model parameters and paths
        k: Number of folds (default: 5)
        stratified: Whether to use stratified k-fold (default: True)

    Returns:
        dict: Best model results with additional metadata
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    # Load data
    df = pd.read_csv(config["PATH"])
    data = IronyDetectionDataset(df)

    # Initialize variables
    fold_results = []
    best_result = None
    best_f1 = -1.0

    # Create directory for current CV session
    cv_path = f"./cv/cv_{TIMESTAMP}_{config["EMB_MODEL"]}"
    os.makedirs(cv_path, exist_ok=True)

    # Choose folding strategy
    if stratified:
        # Extract labels for stratification
        # Assuming dataset has an attribute to get labels
        if hasattr(data, "labels"):
            labels = data.labels
        else:
            raise ValueError(
                "Cannot find labels for stratification. Please ensure your dataset has 'label' or 'target' column, or implement a 'labels' attribute."
            )

        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        splits = kf.split(range(len(data)), labels)  # passing labels for stratification
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        splits = kf.split(range(len(data)))

    print(
        f"Starting {k}-fold {'stratified' if stratified else 'regular'} cross-validation"
    )
    print(f"Results will be saved to: {cv_path}")

    # Perform k-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*50}")
        print(f"Starting fold {fold + 1}/{k}")
        print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

        # Create subsets
        train_subset = torch.utils.data.Subset(data, train_idx)  # pyright: ignore
        val_subset = torch.utils.data.Subset(data, val_idx)  # pyright: ignore

        # Print class distribution if stratified
        if stratified:
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]
            print(
                f"Train class distribution: {dict(zip(*np.unique(train_labels, return_counts=True)))}"
            )
            print(
                f"Val class distribution: {dict(zip(*np.unique(val_labels, return_counts=True)))}"
            )

        try:
            # Run training session
            checkpoint = run_session(
                config=config,
                traind_path=config["PATH"],
                splits={
                    "train": train_subset,
                    "val": val_subset,
                },
            )

            if checkpoint is None:
                print(f"Warning: Failed to retrieve checkpoint for fold {fold + 1}")
                continue

            # Store fold result
            fold_result = {
                "fold": fold,
                "config": config.copy(),
                "metrics": checkpoint.copy(),
                "train_idx": train_idx,
                "val_idx": val_idx,
            }
            fold_results.append(fold_result)

            print(f"Fold {fold + 1} completed - F1: {checkpoint.get('avg_f1', 'N/A')}")

            # Track best result
            current_f1 = checkpoint.get("avg_f1", -1.0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_result = {
                    **checkpoint,
                    "fold": fold,
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                    "config": config.copy(),
                }
                print(f"New best result! F1: {best_f1}")

        except Exception as e:
            print(f"Error in fold {fold + 1}: {str(e)}")
            continue

    # Calculate aggregate statistics
    if fold_results:
        metrics_keys = [
            "avg_acc",
            "avg_f1",
        ]  # metrics stored in the checkpoint returned by run_session
        aggregate_stats = {}

        for metric in metrics_keys:
            values = [
                result["metrics"][metric]
                for result in fold_results
                if metric in result["metrics"]
            ]
            if values:
                aggregate_stats[f"{metric}_mean"] = np.mean(values)
                aggregate_stats[f"{metric}_std"] = np.std(values)
                aggregate_stats[f"{metric}_values"] = values

        # Add aggregate stats to best result
        if best_result:
            best_result["cv_stats"] = aggregate_stats
            best_result["cv_folds"] = len(fold_results)
            best_result["cv_type"] = "stratified" if stratified else "regular"

    # Save results
    if best_result:
        # Save best model
        torch.save(best_result, os.path.join(cv_path, "best_result.pt"))

        # Save all fold results
        torch.save(fold_results, os.path.join(cv_path, "all_folds.pt"))

        # Save summary
        summary = {
            "timestamp": TIMESTAMP,
            "config": config,
            "cv_type": "stratified" if stratified else "regular",
            "n_folds": k,
            "completed_folds": len(fold_results),
            "best_fold": best_result["fold"],
            "best_f1": best_f1,
            "cv_stats": best_result.get("cv_stats", {}),
        }

        with open(os.path.join(cv_path, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n{'='*50}")
        print("Cross-validation completed!")
        print(f"Best F1 score: {best_f1} (fold {best_result['fold'] + 1})")
        if "cv_stats" in best_result:
            print(
                f"Mean F1: {best_result['cv_stats'].get('avg_f1_mean', 'N/A'):.4f} ± {best_result['cv_stats'].get('avg_f1_std', 'N/A'):.4f}"
            )
        print(f"Results saved to: {cv_path}")

        return best_result
    else:
        raise RuntimeError(
            "No successful folds completed. Check your configuration and data."
        )


def main(config):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODELS = ["instructor"]  # "bert-cls", "bert-avg", "sonar",

    for model in MODELS:

        print(
            "\n" * 4,
            "#" * 10,
            f"RUNNING PIPELINE WITH EMBEDDING MODEL: {model}",
            "#" * 10,
            "\n" * 4,
        )

        path = f"results/final_{model}_{TIMESTAMP})_{"A" if not config["STATIC"]["MULTI"] else "B"}/"
        os.makedirs(path, exist_ok=True)
        config["EMB_MODEL"] = model

        best_trial = tune_downstream_model(
            config,
            embedding_model=model,
            num_samples=config["STATIC"]["SAMPLES"],
            bayesian=config["STATIC"]["BAYES"],
        )
        if best_trial is None:
            print("Error in getting the best trial")
            continue

        tuned_config = best_trial.config
        if tuned_config is None:
            continue

        tuned_config["EMB_MODEL"] = model
        print("\n" * 4, "#" * 10, f"STARTING CV: {model}", "#" * 10, "\n" * 4)

        # Post-tuning CV to assess best results
        best_result = kfold_cv(
            config=tuned_config, k=config["STATIC"]["K"], stratified=True
        )
        tuned_config["MAX_EPOCHS"] = best_result["last_epoch"]

        print(
            "\n" * 4,
            "#" * 10,
            f"STARTING TRAINING WITH EMBEDDING MODEL: {model}",
            "#" * 10,
            "\n" * 4,
        )
        classifier_model = IronyClassifier(
            emb_dim=1024 if model == "sonar" else 768,
            n_classes=1 if not config["STATIC"]["MULTI"] else 4,
            h_dim=tuned_config["H_DIM"],
            drop_rate=tuned_config["DROP_RATE"],
            use_learnable_dr=True if tuned_config["REDUX"] else False,
            reduced_dim=tuned_config["REDUX"],
        )
        # Training model with best config
        full_data_training(
            classifier_model,
            tuned_config,
            multiclass=config["STATIC"]["MULTI"],
            data_path=config["STATIC"]["PATH"],
        )

        # Running model on test data and saving results
        test_acc, test_f1 = run_test_data(
            classifier_model,
            path=config["STATIC"]["TEST"],
            emb_model=model,
            device=DEVICE,
            multiclass=config["STATIC"]["MULTI"],
        )

        tuned_config = {
            k: int(v) if isinstance(v, np.int64) else v for k, v in tuned_config.items()
        }
        try:
            with open(
                os.path.join(path, "test_metrics.json"), mode="w", encoding="utf-8"
            ) as json_f:
                json.dump(
                    {**tuned_config, "test_accuracy": test_acc, "test_f1": test_f1},
                    json_f,
                )
        except:
            print("Failed to dump final json")

        del classifier_model

        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print("FINISHED!")


if __name__ == "__main__":

    SMOKE_TEST = {
        "STATIC": {
            "PATH": "./data/train/SemEval2018-T3-train-taskA_emoji.csv",
            "TEST": "./data/test/SemEval2018-T3_gold_test_taskA_emoji.csv",
            "MULTI": False,  # wether to perform binary or multiclass classification
            "OPT": "AdamW",  # tune.choice(["SGD", "ADAM"]),
            "MAX_EPOCHS": 2,
            "SAMPLES": 2,
            "SEED": 42,
            "K": 2,
            "BAYES": True,
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

    CONFIG_A = {
        "STATIC": {
            "PATH": "./data/train/SemEval2018-T3-train-taskA_emoji.csv",
            "TEST": "./data/test/SemEval2018-T3_gold_test_taskA_emoji.csv",
            "MULTI": False,  # wether to perform binary or multiclass classification
            "OPT": "AdamW",  # tune.choice(["SGD", "ADAM"]),
            "MAX_EPOCHS": 20,
            "SAMPLES": 10,
            "SEED": 42,
            "K": 5,
            "BAYES": True,
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

    CONFIG_B = {
        "STATIC": {
            "PATH": "./data/train/SemEval2018-T3-train-taskB_emoji.csv",
            "TEST": "./data/test/SemEval2018-T3_gold_test_taskB_emoji.csv",
            "MULTI": True,  # wether to perform binary or multiclass classification
            "OPT": "AdamW",  # tune.choice(["SGD", "ADAM"]),
            "MAX_EPOCHS": 20,
            "SAMPLES": 10,
            "SEED": 42,
            "K": 5,
            "BAYES": True,
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
    # main(SMOKE_TEST)
    main(CONFIG_A)
    print("\n" * 4, "#" * 20, "\n" * 4)
    main(CONFIG_B)
    


