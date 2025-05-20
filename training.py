import torch

print(torch.__version__)
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from dataset import EmbeddingCollate, IronyDetectionDataset
from explore import count_labels
from model import IronyClassifier

try:
    torch.cuda.manual_seed(42)
except:
    print("CUDA not connected")
import json
import os
from copy import deepcopy
from datetime import datetime

from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm


def train(
    multiclass: bool,
    dataloader,
    model,
    criterion,
    optimizer,
    device,
    scaler,
    l1_coeff=None,
):

    model.train()

    acc_loss = 0.0
    processed = 0
    predictions = []
    targets = []

    for data, label in tqdm(
        dataloader, desc="Training progress", disable=__name__ != "__main__"
    ):
        data = data.to(device)
        label = label.to(device)

        targets.append(label)

        with torch.autocast(
            device_type=device, enabled=(device == "cuda"), dtype=torch.float16
        ):
            logits = model(data)
            penalty = model.l1_penalty(l1_coeff) if l1_coeff else 0.0
            loss = criterion(logits, label) + penalty

        scaler.scale(loss).backward()

        # Unscaling gradients before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if multiclass:
            prob_preds = torch.softmax(logits, dim=1)
            preds = torch.argmax(prob_preds, dim=1)
        else:
            preds = torch.round(torch.sigmoid(logits))

        predictions.append(preds)

        acc_loss += loss.item()
        processed += label.size(0)

    # preds&targets concat + casting to numpy
    predictions = torch.cat(predictions).cpu().detach().numpy()
    targets = torch.cat(targets).cpu().detach().numpy()
    # Computing epoch metrics
    avg_loss = acc_loss / processed

    accuracy = accuracy_score(y_pred=predictions, y_true=targets) * 100

    report_log = classification_report(
        y_pred=predictions,
        y_true=targets,
        zero_division=0,
    )
    avg = "binary" if not multiclass else "weighted"
    f1 = f1_score(y_true=targets, y_pred=predictions, average=avg, zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1": f1,
        "report": report_log,
    }


def eval(
    multiclass: bool,
    val_dataloader,
    model,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    processed = 0
    predictions = []
    targets = []

    for data, label in tqdm(
        val_dataloader, desc="Training progress", disable=__name__ != "__main__"
    ):
        data = data.to(device)
        label = label.to(device)

        if not multiclass:
            targets.append(label.flatten())
        else:
            targets.append(label)

        with torch.no_grad():
            logits = model(data)
            loss = criterion(logits, label)

        if multiclass:
            prob_preds = torch.softmax(logits, dim=1)
            preds = torch.argmax(prob_preds, dim=1)
        else:
            preds = torch.round(torch.sigmoid(logits))

        predictions.append(preds)

        total_loss += loss.item()
        processed += label.size(0)

    predictions = torch.cat(predictions).cpu().detach().numpy()
    targets = torch.cat(targets).cpu().detach().numpy()
    # Computing epoch metrics
    avg_loss = total_loss / processed

    accuracy = accuracy_score(y_pred=predictions, y_true=targets) * 100

    report_log = classification_report(
        y_pred=predictions,
        y_true=targets,
        zero_division=0,
    )
    avg = "binary" if not multiclass else "weighted"
    f1 = f1_score(y_pred=predictions, y_true=targets, average=avg, zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1": f1,
        "report": report_log,
    }


def run_test_data(model, path, device, collate=None, emb_model=None, multiclass=False, batch_size=32):

    if not collate and not emb_model:
        raise TypeError("either pass a collate function or pass and embedding model name")
    df = pd.read_csv(path)
    dataset = IronyDetectionDataset(df)
    if not collate:

        collate = EmbeddingCollate(
            embedding_model=emb_model,
            multiclass=multiclass,
            device=device,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,  # pyright:ignore
    )

    model.eval()

    all_preds = []
    all_labels = []

    for data, label in tqdm(dataloader, desc="Test progress:"):
        data = data.to(device)
        label = label.to(device)
        all_labels.append(label)

        with torch.no_grad():
            logits = model(data)

            if not multiclass:
                preds = torch.round(torch.sigmoid(logits))
            else:
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            all_preds.append(preds)

    all_preds, all_labels = torch.cat(all_preds).cpu().detach().numpy(), torch.cat(all_labels).cpu().detach().numpy()
    accuracy = accuracy_score(all_labels, all_preds)
    avg = "binary" if not multiclass else "weighted"
    f1 = f1_score(all_labels, all_preds, average=avg, zero_division=0)

    print("\n", "=" * 10, " TEST ", "=" * 10, "\n")

    print(classification_report(
        y_true=all_labels,
        y_pred=all_preds,
        zero_division=0
    ))
    print(f"Test accuracy |  {accuracy:.2f}%")
    print(f"Test F1: {f1:.2f}")

    return accuracy, f1




def run_session(
    config: dict,
    traind_path,
    testd_path=None,
    splits: dict | None = None,
    early_stop=True,
    run_test=False,
):
    """
    Running a training session.
    """

    if not traind_path and splits:
        raise TypeError("No data passed through 'traind_path' or 'splits' parameters.")

    DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_SPLIT = 0.8
    if not splits:
        TRAIN_SPLIT = config["SPLIT"]

    LR = config["LR"]
    REDUCED_DIM = config["REDUX"]
    H_DIM = config["H_DIM"]
    EPOCHS = config["MAX_EPOCHS"]
    BATCH_SIZE = config["BATCH_SIZE"]
    DROP_RATE = config["DROP_RATE"]
    L1 = config["L1"]
    WEIGHT_DECAY = config["WEIGHT_DECAY"]
    PATIENCE = config["PATIENCE"]  # how much stagnating epochs to tolerate
    SEED = config["SEED"]

    EMB_MODEL = config["EMB_MODEL"]

    # Creating dir to save best model and config
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    task_indicator = "A" if not config["MULTI"] else "B"
    session_path = f"models/session{task_indicator}_{timestamp}"
    os.makedirs(session_path, exist_ok=True)

    # Setting seed for reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Getting data
    data_df = pd.read_csv(traind_path)
    full_dataset = IronyDetectionDataset(
        data=data_df,
    )
    if splits:  # kfold splitting
        if not isinstance(splits, dict):
            raise ValueError(
                "'cv_splits' has to be a dictionary with 'train' and 'val' keys"
            )
        train_dataset, val_dataset = splits["train"], splits["val"]
    else:  # random split
        train_size = int(TRAIN_SPLIT * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Printing out label distributions
    train_tuples = [full_dataset[idx] for idx in train_dataset.indices]
    train_df = pd.DataFrame(
        data=[(tup[0], tup[1]) for tup in train_tuples],
    )
    train_df.columns = ["Tweet", "Label"]
    print(train_df["Label"].value_counts())

    val_tuples = [full_dataset[idx] for idx in val_dataset.indices]
    val_df = pd.DataFrame(
        data=[(tup[0], tup[1]) for tup in val_tuples],
    )
    val_df.columns = ["Tweet", "Label"]
    print(val_df["Label"].value_counts())

    # Setting up dataloaders
    workers = 0 if DEVICE_NAME == "cuda" else 8

    collate_func = EmbeddingCollate(
        embedding_model=EMB_MODEL, multiclass=config["MULTI"], device=DEVICE_NAME
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=workers,
        collate_fn=collate_func,  # pyright:ignore
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=workers,
        collate_fn=collate_func,  # pyright:ignore
    )

    input_shape = next(iter(train_dataloader))[0].size(
        -1
    )  # getting the embeddings size to determine the model's input dimension

    # Initializing model
    print("Model input dim: ", input_shape)
    model = IronyClassifier(
        emb_dim=input_shape,  # pyright:ignore
        h_dim=H_DIM, drop_rate=DROP_RATE,
        n_classes=1 if not config["MULTI"] else 4,
        use_learnable_dr=REDUCED_DIM is not None,
        reduced_dim=REDUCED_DIM,
    )


    model.to(DEVICE_NAME)

    scaler = torch.GradScaler("cuda", enabled=(DEVICE_NAME == "cuda"))

    if not config["MULTI"]:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        train_labels = [full_dataset.labels[i] for i in train_dataset.indices]

        label_series = pd.Series(train_labels)
        label_counts = label_series.value_counts().sort_index().to_numpy()        
        weights = 1.0 / label_counts # inverse frequency
        weights = weights / weights.sum()  # Normalize
        weights_tensor = torch.FloatTensor(weights).to(DEVICE_NAME)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)

    if config["OPT"] == "SGD":
        optim = torch.optim.SGD(
            params=model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
    else:
        optim = torch.optim.AdamW(
            params=model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

    plat_scheduler = ReduceLROnPlateau(
        optimizer=optim,
        mode="max",
        patience=PATIENCE,
        factor=0.1,
    )

    # Variables used in early stopping (based on f1 performance)
    best_f1 = -1.0
    tolerance = max(3, PATIENCE)
    best_model = None
    all_f1 = 0.0
    all_acc = 0.0
    last_epoch = 0

    try:
        for epoch in range(EPOCHS):

            print("\n", "-" * 40, "\n")
            print("#" * 10, f" EPOCH: {epoch+1} ", "#" * 10, "\n")

            train_result = train(
                multiclass=config["MULTI"],
                dataloader=train_dataloader,
                model=model,
                criterion=loss_fn,
                optimizer=optim,
                device=DEVICE_NAME,
                scaler=scaler,
                l1_coeff=L1,
            )

            print("\n", "*" * 10, "TRAIN METRICS", "*" * 10)
            print()
            print(train_result["report"])
            print(
                f"Running Loss | {train_result["loss"]:.5f}\nTrain Accuracy | {train_result["accuracy"]:.2f}%"
            )

            val_result = eval(
                multiclass=config["MULTI"],
                val_dataloader=val_dataloader,
                model=model,
                criterion=loss_fn,
                device=DEVICE_NAME,
            )
            val_f1 = val_result["f1"]
            print("\n", "*" * 10, "VALIDATION METRICS", "*" * 10)
            print()
            print(val_result["report"])
            print(
                f"Validation loss | {val_result["loss"]:.5f}\nValidation Accuracy | {val_result["accuracy"]:.2f}%"
            )
            all_f1 += val_result["f1"]
            all_acc += val_result["accuracy"]
            last_epoch = epoch + 1

            # LR scheduling
            plat_scheduler.step(val_f1)

            if not early_stop:
                continue

            # Early stopping
            if epoch != 0 and val_f1 < best_f1:

                tolerance -= 1

                if tolerance == 0:
                    print("\nEXITING EARLY...\n")
                    break  # Exiting loop

                continue

            # Reset tolerance and save best result
            tolerance = max(3, PATIENCE)

            best_f1 = val_f1
            best_model = deepcopy(model.state_dict())

    except KeyboardInterrupt:
        print("\nSession stopped early by SIGINT\n")

    model.load_state_dict(best_model)

    # Saving relevant data for current session
    checkpoint = None
    if best_model:
        checkpoint = {
            "model": model,
            "config": config,
            "avg_acc": all_acc / last_epoch,
            "avg_f1": all_f1 / last_epoch,
            "last_epoch": last_epoch
        }

        torch.save(
            checkpoint,
            os.path.join(session_path, "best_model.pt"),
        )

    if run_test == True:
        test_acc, test_f1 = run_test_data(
            model=model,
            path=testd_path,
            device=DEVICE_NAME,
            collate=collate_func,
            batch_size=BATCH_SIZE,
            multiclass=config["MULTI"]
        )
        with open(os.path.join(session_path, "test_metrics.pt"), mode="w", encoding="utf-8") as json_f:
            json.dump({
                "test_acc": test_acc,
                "test_f1": test_f1,
                "last_epoch": last_epoch,
            }, json_f)

    return checkpoint


if __name__ == "__main__":

    config = {
        "MULTI": True,
        "EMB_MODEL": "bert-cls",
        "OPT": "AdamW",

        "SEED": 42,
        "SPLIT": 0.8,
        "LR": 1e-4,
        "REDUX": 256,
        "H_DIM": 64,
        "MAX_EPOCHS": 5,
        "BATCH_SIZE": 32,
        "DROP_RATE": 0.4,
        "L1": 1e-5,
        "WEIGHT_DECAY": 1e-4,
        "PATIENCE": 3,  # how much stagnating epochs to tolerate
    }

    check = run_session(
        config,
        traind_path="./data/train/SemEval2018-T3-train-taskB_emoji.csv",
        testd_path="./data/test/SemEval2018-T3_gold_test_taskB_emoji.csv",
        run_test=True,
    )
