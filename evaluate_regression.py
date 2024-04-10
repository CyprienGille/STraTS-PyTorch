from typing import Tuple

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
    n_vars=29,
    var_id=0,
    normalize_times=True,
    normalize_vars=True,
    verbose=False,
    n_layers=5,
    dim_embed=104,
    n_heads=4,
) -> Tuple[STraTS, DataLoader]:
    """Get a trained model and the corresponding testing dataloader

    Parameters
    ----------
    model_dir : str
        The path to the model dir, for example 'exp_creat_reg/exp_14/'
    data_path : str
        The path to the preprocessed data
    back_interval : int, optional
        The observation window width in minutes, by default 2880
    n_vars : int, optional
        The number of features in the data, by default 29
    var_id : int, optional
        The id of the target variable, by default 0
    normalize_times : bool, optional
        Whether to normalize times, by default True
    normalize_vars : bool, optional
        Whether to normalize variables, by default True
    verbose : bool, optional
        Whether to display progress bars, by default False
    n_layers : int, optional
        The number of transformer layers in the model, by default 5
    dim_embed : int, optional
        The embedding dimension of the model, by default 104
    n_heads : int, optional
        The number of heads in the model, by default 4

    Returns
    -------
    Tuple[STraTS, DataLoader]
        The model with its loaded weights, and the test DataLoader
    """
    ds = MIMIC_Reg(data_path, var_id=var_id, crop_back_interval=back_interval)
    ds.restrict_to_indexes(np.load(model_dir + "test_idx.npy"))
    ds.normalize(normalize_vars, normalize_times, verbose=verbose)
    dl = DataLoader(ds, batch_size=1, collate_fn=padded_collate_fn, shuffle=False)

    model = STraTS(
        n_var_embs=n_vars,
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
    # Note : bottom-up variables -> creat=0
    # Note : top-down variables -> creat=47
    var_id = 47

    model, test_dl = get_model_and_dl(
        model_dir=path,
        # data_path="generated/29var_EH_culled_reg.csv",
        data_path="generated/top_206_culled_reg.csv",
        n_vars=206,
        var_id=var_id,
        back_interval=2880,
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
    creat_mean = test_dl.dataset.means[var_id]
    creat_std = test_dl.dataset.stds[var_id]
    for demog, values, times, variables, tgt, _, masks in tqdm(test_dl):
        pred_val = model(demog, values, times, variables, masks)
        error += loss_fn(pred_val, tgt.squeeze()).item()

        pred_val_denorm = denorm(pred_val.item(), creat_mean, creat_std)
        tgt_val_denorm = denorm(tgt.item(), creat_mean, creat_std)
        y_pred.append(creat_to_4_stages(pred_val_denorm))
        y_true.append(creat_to_4_stages(tgt_val_denorm))

    print(f"Average Test Mean Error for exp {exp_n} : {error / len(test_dl):.6f}")
    print_metrics(y_true, y_pred)
