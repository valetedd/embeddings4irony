import torch

print(torch.__version__)
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import IronyDetectionDataset
from models import TaskAClassifier

# XXX: import sklearn for metrics


def set_dataloader(data: torch.utils.data.Dataset, **kwargs):
    pass


def train(
    data,
    model,
    criterion,
    optimizer,
    device,
    scaler=None,
):
    
    model.train()

    acc_loss = 0.0
    correct = 0
    processed = 0
    

    for data, label in data:
        
        data = data.to(device)
        if scaler:
            with torch.autocast(device):
                logits = model(data)
                loss = criterion(logits, label)

            scaler.scale(loss).backwards()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()
            
        else:
            logits = model(data)

            optimizer.zero_grad()
            
            loss = criterion(logits, label)
            loss.backwards()

            optimizer.step()

            preds = torch.sigmoid(logits)

        acc_loss += loss
        correct += (preds == label).sum().item() 
        processed += label.size(0)

    
    avg_loss = acc_loss / processed
    accuracy = correct * 100 / processed

    return avg_loss, accuracy


def eval(
    val_data,
    model,
    criterion,
    device,
):

    model.eval()
    
    total_loss = 0.0
    correct = 0
    processed = 0

    for data, label in val_data:
        data = data.to(device)

        with torch.inference_mode():
            logits = model(data)
            loss = criterion(data, label)
            preds = torch.sigmoid(logits)

        total_loss += loss
        correct += (preds == label).sum().item() 
        processed += data.size(0)

    avg_loss = total_loss / processed
    accuracy = correct * 100 / processed

    return avg_loss, accuracy


def test():

    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
    model = "bert-cls"
    LR = 1e-3
    H_DIM = 512
    EPOCHS = 10
    

    data_df = pd.read_csv("data/train/SemEval2018-T3-train-taskA.csv")
    dataset = IronyDetectionDataset(
        data=data_df,
        embedding_model=model, # pyright: ignore
        device=DEVICE,
    )
    dataset.embed_data()
    model = TaskAClassifier(
        emb_dim=dataset.embeddings[0].size(0), 
        h_dim=H_DIM,
        n_classes=1
    )

    dataloader = set_dataloader(data=dataset)
    scaler = None
    scaler = torch.amp.GradScaler(device=DEVICE) if DEVICE == "gpu" else None # pyright:ignore
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(
        params=model.parameters(),
        lr=LR,
    )

    for epoch in range(EPOCHS):
        avg_loss, acc = train(
            data=dataloader,
            model=model,
            criterion=loss_fn,
            optimizer=optim,
            device=DEVICE,
            scaler=scaler,
        )













