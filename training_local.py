# Imports
import os
import numpy as np
from tqdm import tqdm
from typing import Optional
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from models.strats import STraTS
from dataset_int_classif import MIMIC_Int_Classif, padded_collate_fn

# Hyperparameters
exp_dir = "exp_creatinine/"
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

exp_n = len(os.listdir(exp_dir))
saving_path = f"{exp_dir}exp_{exp_n}/"
# make the experiment directory and fail if there is already an experiment with that number
os.makedirs(saving_path, exist_ok=False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

architecture = STraTS
train_prop = 0.8
train_batch_size = 20
test_batch_size = 20
learning_rate = 0.0001
num_epochs = 1
dropout = 0.2
n_layers = 3
dim_embed = 104
n_heads = 4
weight_class_0 = 1.33
weight_class_1 = 7.5
weight_class_2 = 21.0
weight_class_3 = 22.0
normalize_vars = True
normalize_times = False
sched_patience = 2


# Dataset
train_ds = MIMIC_Int_Classif(
    data_path="generated/creat17NoText_culled.csv",
)
test_ds = MIMIC_Int_Classif(
    data_path="generated/creat17NoText_culled.csv",
)
# Randomly split into train/test datasets
n_stays = len(train_ds)
train_indexes, test_indexes = train_test_split(
    list(range(n_stays)), train_size=train_prop
)
train_ds.restrict_to_indexes(train_indexes)
test_ds.restrict_to_indexes(test_indexes)
train_ds.normalize(
    normalize_vars=normalize_vars, normalize_times=normalize_times, verbose=True
)
test_ds.normalize(
    normalize_vars=normalize_vars, normalize_times=normalize_times, verbose=True
)


# save the training/testing indexes for inference
np.save(f"{saving_path}train_idx.npy", train_indexes)
np.save(f"{saving_path}test_idx.npy", test_indexes)


# DataLoaders
train_dl = DataLoader(
    dataset=train_ds,
    batch_size=train_batch_size,
    shuffle=True,
    collate_fn=padded_collate_fn,
    drop_last=True,
)
test_dl = DataLoader(
    dataset=test_ds,
    batch_size=test_batch_size,
    shuffle=False,
    collate_fn=padded_collate_fn,
    drop_last=True,
)


# Model
model = architecture(
    n_var_embs=17,
    dim_demog=2,
    dim_embed=dim_embed,
    n_layers=n_layers,
    n_heads=n_heads,
    dropout=dropout,
    forecasting=False,
    n_classes=4,
)
model = model.to(DEVICE)

# Optimizer
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Loss function (Weighted CE)
weight = torch.Tensor(
    [weight_class_0, weight_class_1, weight_class_2, weight_class_3]
).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
# LR Scheduler
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, patience=sched_patience, verbose=True
)


# Training step
def train_one_epoch(
    model: STraTS,
    train_dl: DataLoader,
    optim: torch.optim.Optimizer,
    loss_fn,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    """Note: only provide a scheduler if its step must be called after each batch."""
    avg_train_loss = 0

    # Training mode
    model.train()

    pbar = tqdm(train_dl)

    for demog, values, times, variables, tgt, masks in pbar:
        optim.zero_grad()

        demog = demog.to(DEVICE)
        values = values.to(DEVICE)
        times = times.to(DEVICE)
        variables = variables.to(DEVICE)
        tgt = tgt.type(torch.LongTensor).to(DEVICE)
        masks = masks.to(DEVICE)

        pred = model(demog, values, times, variables, masks)

        loss = loss_fn(pred, tgt)
        loss_item = loss.item()
        pbar.set_description(f"Current loss: {loss_item:.6f}")
        avg_train_loss += loss_item
        loss.backward()
        # clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        if scheduler is not None:
            scheduler.step()
    return avg_train_loss / len(train_dl)


# Testing step
def test_model(
    model: STraTS,
    test_dl: DataLoader,
    loss_fn,
):
    avg_test_loss = 0

    # Testing mode
    model.eval()

    for (
        demog,
        values,
        times,
        variables,
        tgt,
        masks,
    ) in tqdm(test_dl):
        demog = demog.to(DEVICE)
        values = values.to(DEVICE)
        times = times.to(DEVICE)
        variables = variables.to(DEVICE)
        tgt = tgt.type(torch.LongTensor).to(DEVICE)
        masks = masks.to(DEVICE)

        pred = model(demog, values, times, variables, masks)

        loss = loss_fn(pred, tgt)
        avg_test_loss += loss.item()
    return avg_test_loss / len(test_dl)


# Main part of the script
best_test_loss = test_model(model, test_dl, loss_fn=loss_fn)
print(f"Starting test loss: {best_test_loss:.6f}")

for epoch in range(num_epochs):
    avg_train_loss = train_one_epoch(model, train_dl, optim, loss_fn=loss_fn)
    avg_test_loss = test_model(model, test_dl, loss_fn=loss_fn)
    print(
        f"Training: Epoch {epoch+1:^4} --- Training Loss={avg_train_loss:.6f} --- Testing Loss={avg_test_loss:.6f}\n"
    )

    if avg_test_loss < best_test_loss:
        # if best model yet, save the weights
        best_test_loss = avg_test_loss
        torch.save(model.state_dict(), f"{saving_path}STraTS.pth")

    sched.step(avg_test_loss)
