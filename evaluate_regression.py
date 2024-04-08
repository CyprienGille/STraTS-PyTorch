import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from culling import creat_to_4_stages
from evaluate_int_classif import print_metrics
from strats_pytorch.datasets.dataset_regression import MIMIC_Reg, padded_collate_fn
from strats_pytorch.models.strats import STraTS
from strats_pytorch.utils import denorm


def get_model_and_dl(
    model_dir: str,
    data_path: str,
    back_interval=2880,
    normalize_times=True,
    normalize_vars=True,
    verbose=False,
    n_layers=5,
    dim_embed=104,
    n_heads=4,
):
    ds = MIMIC_Reg(data_path, var_id=47, crop_back_interval=back_interval)
    ds.restrict_to_indexes(np.load(model_dir + "test_idx.npy"))
    ds.normalize(normalize_vars, normalize_times, verbose=verbose)
    dl = DataLoader(ds, batch_size=1, collate_fn=padded_collate_fn, shuffle=False)

    model = STraTS(
        n_var_embs=206,
        dim_demog=2,
        dropout=0.0,
        activation="gelu",
        n_layers=n_layers,
        dim_embed=dim_embed,
        n_heads=n_heads,
        forecasting=False,
        regression=True,
    )

    model.load_state_dict(torch.load(model_dir + "STraTS.pth"))

    return model, dl


if __name__ == "__main__":
    exp_n = 52
    path = f"exp_creat_reg/exp_{exp_n}/"

    model, test_dl = get_model_and_dl(
        model_dir=path,
        data_path="generated/top_206_culled_reg.csv",
        n_layers=2,
        dim_embed=102,
        n_heads=3,
    )
    model.eval()

    loss_fn = torch.nn.L1Loss()

    print("Getting metrics...")
    error = 0.0
    y_pred = []
    y_true = []
    creat_mean = test_dl.dataset.means[47]
    creat_std = test_dl.dataset.stds[47]
    for demog, values, times, variables, tgt, _, masks in tqdm(test_dl):
        pred_val = model(demog, values, times, variables, masks)
        error += loss_fn(pred_val, tgt.squeeze()).item()

        pred_val_denorm = denorm(pred_val.item(), creat_mean, creat_std)
        tgt_val_denorm = denorm(tgt.item(), creat_mean, creat_std)
        y_pred.append(creat_to_4_stages(pred_val_denorm))
        y_true.append(creat_to_4_stages(tgt_val_denorm))

    print(f"Average Test Mean Error for exp {exp_n} : {error / len(test_dl):.6f}")
    print_metrics(y_true, y_pred)
