# Imports
import os
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from strats_pytorch.datasets.dataset_regression import MIMIC_Reg, padded_collate_fn
from strats_pytorch.models.strats import STraTS_Dense

# Hyperparameters
exp_dir = "exp_creat_reg/"
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
exp_n = len(os.listdir(exp_dir))
saving_path = f"{exp_dir}exp_{exp_n}/"
# make the experiment directory and fail if there is already an experiment with that number
os.makedirs(saving_path, exist_ok=False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = {
    "name": "STraTS_29-BUN",
    "train_prop": 0.8,
    "train_batch_size": 10,
    "test_batch_size": 10,
    "learning_rate": 0.00025,
    "num_epochs": 25,
    "dropout": 0.1,
    "n_layers": 2,
    "dim_embed": 104,
    "n_heads": 4,
    "normalize_vars": True,
    "normalize_time": True,
    "back_interval": 7000,
}

data_path = "generated/29var_culled_reg.csv"
# data_path = "generated/top_206_culled_reg.csv"

# Dataset
# Note : bottom-up -> creat is 0, BUN is 18
# Note : top-down -> creat_id is 47, BUN is 49
# Note : 1d, 2d, 3d, 4d, 5d = 1440, 2880, 4320, 5760, 7200 minutes
train_ds = MIMIC_Reg(
    # data_path="generated/29var_culled_reg.csv",
    data_path=data_path,
    keep_tgt_var=False,
    var_id=0,
    crop_back_interval=cfg["back_interval"],
    masked_features=[18],
)
test_ds = MIMIC_Reg(
    # data_path="generated/29var_culled_reg.csv",
    data_path=data_path,
    keep_tgt_var=False,
    var_id=0,
    crop_back_interval=cfg["back_interval"],
    masked_features=[18],
)
n_stays = len(train_ds)
train_indexes, test_indexes = train_test_split(
    list(range(n_stays)), train_size=cfg["train_prop"]
)
train_ds.restrict_to_indexes(train_indexes)
test_ds.restrict_to_indexes(test_indexes)
train_ds.normalize(
    normalize_vars=cfg["normalize_vars"],
    normalize_times=cfg["normalize_time"],
    verbose=True,
)
test_ds.normalize(
    normalize_vars=cfg["normalize_vars"],
    normalize_times=cfg["normalize_time"],
    verbose=True,
)


# save the training/testing indexes for inference
np.save(f"{saving_path}train_idx.npy", train_ds.indexes)
np.save(f"{saving_path}test_idx.npy", test_ds.indexes)


# DataLoaders
train_dl = DataLoader(
    dataset=train_ds,
    batch_size=cfg["train_batch_size"],
    shuffle=True,
    collate_fn=padded_collate_fn,
    drop_last=True,
)
test_dl = DataLoader(
    dataset=test_ds,
    batch_size=cfg["test_batch_size"],
    shuffle=False,
    collate_fn=padded_collate_fn,
    drop_last=True,
)


# Model
model = STraTS_Dense(
    n_var_embs=29,
    dim_demog=2,
    dim_embed=cfg["dim_embed"],
    n_layers=cfg["n_layers"],
    n_heads=cfg["n_heads"],
    dropout=cfg["dropout"],
    activation="gelu",
    forecasting=False,
    regression=True,
)
model = model.to(DEVICE)
# Optimizer
optim = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
loss_fn = torch.nn.HuberLoss(delta=1.0)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, factor=0.5, patience=3, verbose=True
)


# Training step
def train_one_epoch(
    model,
    train_dl: DataLoader,
    optim: torch.optim.Optimizer,
    loss_fn,
):
    avg_train_loss = 0
    # Training mode
    model.train()

    pbar = tqdm(train_dl)

    for demog, values, times, variables, tgt, _, masks in pbar:
        optim.zero_grad()

        demog = demog.to(DEVICE)
        values = values.to(DEVICE)
        times = times.to(DEVICE)
        variables = variables.to(DEVICE)
        tgt = tgt.to(DEVICE)
        masks = masks.to(DEVICE)
        pred = model(demog, values, times, variables, masks)

        loss = loss_fn(pred, tgt)
        loss_item = loss.item()
        pbar.set_description(f"Current loss: {loss_item:.6f}")
        avg_train_loss += loss_item
        loss.backward()
        optim.step()

    return avg_train_loss / len(train_dl)


# Testing step
def test_model(
    model,
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
        _,
        masks,
    ) in tqdm(test_dl):
        demog = demog.to(DEVICE)
        values = values.to(DEVICE)
        times = times.to(DEVICE)
        variables = variables.to(DEVICE)
        tgt = tgt.to(DEVICE)
        masks = masks.to(DEVICE)

        pred = model(demog, values, times, variables, masks)

        loss = loss_fn(pred, tgt)
        avg_test_loss += loss.item()

    return avg_test_loss / len(test_dl)


# Main part of the script
print(f"=== Exp {exp_n} ({cfg['name']}) ===")
print("Computing starting test loss...")
best_test_loss = test_model(model, test_dl, loss_fn=loss_fn)
epoch_best_loss = -1
running_lr = [cfg["learning_rate"]]
print(f"Starting test loss: {best_test_loss:.6f}")

for epoch in range(cfg["num_epochs"]):
    avg_train_loss = train_one_epoch(model, train_dl, optim, loss_fn=loss_fn)
    avg_test_loss = test_model(model, test_dl, loss_fn=loss_fn)
    print(
        f"Training: Epoch {epoch+1:^4} --- Training Loss={avg_train_loss:.6f} --- Testing Loss={avg_test_loss:.6f}\n"
    )

    if avg_test_loss < best_test_loss:
        # if best model yet, save the weights
        best_test_loss = avg_test_loss
        epoch_best_loss = epoch
        torch.save(model.state_dict(), f"{saving_path}STraTS.pth")

    sched.step(avg_test_loss)
    current_lr = sched.get_last_lr()
    if current_lr != running_lr:
        print(f"Adjusting lr from {running_lr} to {current_lr}.")
        running_lr = current_lr

print(
    f"=== End of exp {exp_n} ({cfg['name']}) - Best epoch : {epoch_best_loss+1}/{cfg['num_epochs']} ==="
)
