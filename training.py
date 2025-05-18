import torch

print(torch.__version__)
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from dataset import EmbeddingCollate, IronyDetectionDataset
from models import TaskAClassifier

try:
    torch.cuda.manual_seed(42)
except:
    print("CUDA not connected")
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

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

    for data, label in tqdm( dataloader, desc="Training progress" , disable=__name__ != "__main__"):
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

    for data, label in tqdm( val_dataloader, desc="Training progress" , disable=__name__ != "__main__"):        
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


def run_test_data(model, path, device, collate, multiclass=False, batch_size=32):

    df = pd.read_csv(path)
    dataset = IronyDetectionDataset(df)
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
        all_labels.extend(label.detach().cpu().numpy())

        with torch.no_grad():
            logits = model(data)
            if not multiclass:
                preds = torch.round(torch.sigmoid(logits))
            else:
                preds = torch.argmax(torch.softmax(logits, dim=1))
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    avg = "binary" if not multiclass else "weighted"
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print("\n", "=" * 10, " TEST ", "=" * 10, "\n")

    print(f"Test accuracy |  {accuracy:.2f}%")
    print(f"Test F1: {f1:.2f}") 

    return accuracy, f1


def kfold_cv(config, k=5):
    """
    Function to perform k-fold cross validation.
    """

    from sklearn.model_selection import KFold

    df = pd.read_csv("./data/train/SemEval2018-T3-train-taskA_emoji.csv")
    data = IronyDetectionDataset(df)
    best_result = {}
    # Creating a dir for current cv sessione
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    path = Path(f"./cv/cv_{timestamp}")
    os.makedirs(path)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        # Creating subsets
        train_subset = torch.utils.data.Subset(data, train_idx)
        val_subset = torch.utils.data.Subset(data, val_idx)
        print(f"Starting fold n.{fold}")
        # Running training session
        checkpoint = run_session(
            config=config,
            multiclass=config["MULTI"],
            traind_path="./data/train/SemEval2018-T3-train-taskA_emoji.csv",
            testd_path="./data/test/SemEval2018-T3_gold_test_taskA_emoji.csv",
            splits={
                "train": train_subset,
                "val": val_subset,
            },
        )
        if checkpoint is None:
            raise ValueError("Failed to retrive checkpoint")

        print(f"\n\nFinished fold {fold} with the following configuration:\n{config}")

        if best_result and checkpoint["avg_f1"] < best_result["avg_f1"]:
            continue

        best_result = checkpoint
        best_result["train"] = train_subset
        best_result["val"] = val_subset

    torch.save(best_result, path / "best_result.pt")
    return best_result


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

    DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_SPLIT = config["SPLIT"]

    LR = config["LR"]
    REDUCED_DIM = config["REDUX"]
    H_DIM = config["H_DIM"]
    EPOCHS = config["EPOCHS"]
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

    collate_func = EmbeddingCollate(embedding_model=EMB_MODEL, multiclass=config["MULTI"], device=DEVICE_NAME)

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
    model = TaskAClassifier(
        emb_dim=input_shape,  # pyright:ignore
        h_dim=H_DIM,
        drop_rate=DROP_RATE,
        n_classes=1 if not config["MULTI"] else 4,
        use_learnable_dr=REDUCED_DIM is not None,
        reduced_dim=REDUCED_DIM,
        use_attention=config["USE_ATT"],
    )
    model.to(DEVICE_NAME)

    scaler = torch.GradScaler("cuda", enabled=(DEVICE_NAME == "cuda"))

    if not config["MULTI"]:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

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

    # Variables used in early stopping
    best_loss = 0.0
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
                multiclass=True if config["MULTI"] else False,
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
                multiclass=True if config["MULTI"] else False,
                val_dataloader=val_dataloader,
                model=model,
                criterion=loss_fn,
                device=DEVICE_NAME,
            )
            val_loss = val_result["loss"]
            print("\n", "*" * 10, "VALIDATION METRICS", "*" * 10)
            print()
            print(val_result["report"])
            print(
                f"Validation loss | {val_loss:.5f}\nValidation Accuracy | {val_result["accuracy"]:.2f}%"
            )
            all_f1 += val_result["f1"]
            all_acc += val_result["accuracy"]
            last_epoch = epoch + 1

            if epoch >= EPOCHS // 2:
                plat_scheduler.step(val_loss)
            else:
                scheduler.step()

            if not early_stop:
                continue

            # Early stopping
            if epoch != 0 and val_loss > best_loss:

                tolerance -= 1

                if tolerance == 0:
                    print("\nEXITING EARLY...\n")
                    break  # Exiting loop

                continue

            # Reset tolerance and save best result
            tolerance = max(3, PATIENCE)

            best_loss = val_loss
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
        }

        torch.save(
            checkpoint,
            os.path.join("best_model.pt"),
        )

    if run_test == True:
        run_test_data(
            model=model,
            path=testd_path,
            device=DEVICE_NAME,
            collate=collate_func,
            batch_size=BATCH_SIZE,
        )
    return checkpoint


if __name__ == "__main__":

    config = {
        "EMB_MODEL": "sonar",
        "SEED": 42,
        "SPLIT": 0.8,
        "LR": 1e-4,
        "REDUX": 256,
        "H_DIM": 64,
        "EPOCHS": 30,
        "BATCH_SIZE": 32,
        "DROP_RATE": 0.4,
        "L1": 1e-5,
        "WEIGHT_DECAY": 1e-4,
        "PATIENCE": 3,  # how much stagnating epochs to tolerate
        "USE_ATT": True,
        "OPT": "AdamW",

        "MULTI": False
    }

    check = run_session(
        config,
        traind_path="./data/train/SemEval2018-T3-train-taskA_emoji.csv",
        testd_path="./data/test/SemEval2018-T3_gold_test_taskA_emoji.csv",
        run_test=True,
    )
