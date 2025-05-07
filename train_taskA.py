import torch

torch.manual_seed(42)
print(torch.__version__)

import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from dataset import EmbeddingCollate, IronyDetectionDataset
from models import TaskAClassifier

try:
    torch.cuda.manual_seed(42)
except:
    print("CUDA not connected")
from tqdm import tqdm

# XXX: import sklearn for metrics


def train(
    dataloader,
    model,
    criterion,
    optimizer,
    device,
    scaler=None,
    scheduler=None,
):

    model.train()

    acc_loss = 0.0
    correct = 0
    processed = 0

    for data, label in tqdm(dataloader, desc="Training Progress"):
        data = data.to(device)
        label = label.to(device)

        if scaler != None:
            with torch.autocast(device):
                logits = model(data)
                loss = criterion(logits, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

        else:
            logits = model(data)

            loss = criterion(logits, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        if scheduler:
            scheduler.step()

        preds = torch.round(torch.sigmoid(logits))

        acc_loss += loss.item() * label.size(0)
        correct += (preds == label).sum().item()
        processed += label.size(0)

    avg_loss = acc_loss / processed
    accuracy = correct * 100 / processed

    return avg_loss, accuracy


def eval(
    val_dataloader,
    model,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    correct = 0
    processed = 0

    for data, label in tqdm(val_dataloader, desc="Validation progress:"):
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            logits = model(data)
            loss = criterion(logits, label)
            preds = torch.round(torch.sigmoid(logits))

        total_loss += loss.item() * label.size(0)
        correct += (preds == label).sum().item()
        processed += label.size(0)

    avg_loss = total_loss / processed
    accuracy = correct * 100 / processed

    return avg_loss, accuracy


def test_run():

    DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_SPLIT = 0.8

    LR = 5e-5
    H_DIM = 128
    EPOCHS = 20
    BATCH_SIZE = 32
    DROP_RATE = 0.4
    PATIENCE = 3  # how much stagnating epochs to tolerate

    EMB_MODEL = "sonar"
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    data_df = pd.read_csv("data/train/SemEval2018-T3-train-taskA.csv")
    full_dataset = IronyDetectionDataset(
        data=data_df,
    )

    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    workers = 0 if DEVICE_NAME == "cuda" else 8

    collate_func = EmbeddingCollate(embedding_model=EMB_MODEL, device=DEVICE_NAME)

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
    print("Model input dim: ", input_shape)  # pyright:ignore
    model = TaskAClassifier(
        emb_dim=input_shape,  # pyright:ignore
        h_dim=H_DIM,
        n_classes=1,
        drop_rate=DROP_RATE,
    )
    model.to(DEVICE_NAME)

    scaler = torch.GradScaler("cuda") if DEVICE_NAME == "cuda" else None
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(
        params=model.parameters(),
        lr=LR,
    )

    scheduler = ReduceLROnPlateau(
        optimizer=optim,
        mode="min",
        patience=3,
    )

    # Two variables used in early stopping
    best_loss = 0
    tolerance = PATIENCE
    best_model = None
    for epoch in range(EPOCHS):

        print("\n", "-" * 40, "\n")
        print("#" * 5, f" EPOCH: {epoch+1} ", "#" * 5, "\n")

        avg_loss, train_acc = train(
            dataloader=train_dataloader,
            model=model,
            criterion=loss_fn,
            optimizer=optim,
            device=DEVICE_NAME,
            scaler=scaler,
        )

        print("\n", "*" * 10, "TRAIN METRICS", "*" * 10)
        print(f"Running Loss | {avg_loss:.5f}\nTrain Accuracy | {train_acc:.2f}%")

        val_loss, val_acc = eval(
            val_dataloader=val_dataloader,
            model=model,
            criterion=loss_fn,
            device=DEVICE_NAME,
        )
        print("\n", "*" * 10, "VALIDATION METRICS", "*" * 10)
        print(f"Validation loss | {val_loss:.5f}\nValidation Accuracy | {val_acc:.2f}%")

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)

        # Early stopping
        if val_loss > best_loss:

            if tolerance == 0:
                print("Exiting early...")
                break  # Exiting loop

            tolerance -= 1
            continue

        best_loss = val_loss
        best_model = model.state_dict().copy()

    if best_model:
        torch.save(best_model, "models/best_model.pt")

    test_accuracy = run_test_data(
        model=model,
        path="./data/test/SemEval2018-T3_input_test_taskA.csv",
        device=DEVICE_NAME,
    )
    print("=" * 10, " TEST ", "=" * 10)
    print("Test accuracy | ", test_accuracy)


def run_test_data(
    model,
    path,
    device,
    batch_size=32,
):

    df = pd.read_csv(path)
    dataset = IronyDetectionDataset(df)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
    )

    model.eval()

    correct = 0
    processed = 0

    for data, label in tqdm(dataloader, desc="Validation progress:"):
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            logits = model(data)
            preds = torch.round(torch.sigmoid(logits))

        correct += (preds == label).sum().item()
        processed += label.size(0)

    accuracy = correct * 100 / processed

    return accuracy


if __name__ == "__main__":
    test_run()
