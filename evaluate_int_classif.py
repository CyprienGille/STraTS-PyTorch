import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from strats_pytorch.datasets.dataset_int_classif import (
    MIMIC_Int_Classif,
    padded_collate_fn,
)
from strats_pytorch.models.strats import STraTS


def get_model_and_dl(
    model_dir: str,
    data_path: str,
    normalize_times=True,
    normalize_vars=True,
    verbose=False,
    n_layers=5,
    dim_embed=104,
    n_heads=4,
):
    """Initialize a model and DataLoader for inference:

    The DataLoader will have batch size 1 and the model will load its state dict.
    """
    ds = MIMIC_Int_Classif(data_path)
    ds.restrict_to_indexes(np.load(model_dir + "test_idx.npy"))
    ds.normalize(normalize_vars, normalize_times, verbose=verbose)
    dl = DataLoader(ds, batch_size=1, collate_fn=padded_collate_fn, shuffle=False)

    model = STraTS(
        n_var_embs=29,
        dim_demog=2,
        dropout=0.0,
        n_layers=n_layers,
        dim_embed=dim_embed,
        n_heads=n_heads,
        forecasting=False,
        regression=False,
        n_classes=4,
    )

    model.load_state_dict(torch.load(model_dir + "STraTS.pth"))

    return model, dl


def get_y_true_y_probs(
    model: STraTS,
    test_dl: DataLoader,
):
    """Note: returns y_true and y_probs as numpy arrays, ala scikit-learn"""
    y_true = []
    y_probs = []
    softmax = torch.nn.Softmax(dim=-1)
    # Testing mode
    model.eval()
    for demog, values, times, variables, tgt, masks in tqdm(test_dl):
        pred = model(demog, values, times, variables, masks)

        y_true.append(tgt.item())
        y_probs.append(softmax(pred).numpy(force=True))
    return np.array(y_true), np.array(y_probs)


def print_metrics(y_true, y_pred, y_probs=None):
    """Prints:

    - The number of samples per class in y_true
    - The number of samples per class in y_pred
    - The number of correct predictions per class in y_pred
    - The proportion of samples in each class in y_true
    - The proportion of samples in each class in y_pred
    - The classification accuracy
    - If y_probs is not None, the One-versus-one ROC-AUC
    """

    true_per_class = {0: 0, 1: 0, 2: 0, 3: 0}
    pred_per_class = {0: 0, 1: 0, 2: 0, 3: 0}
    correct_per_class = {0: 0, 1: 0, 2: 0, 3: 0}
    accuracy = 0
    for i, elem in enumerate(y_true):
        true_per_class[elem] += 1

        pred_elem = y_pred[i]
        pred_per_class[pred_elem] += 1

        if elem == pred_elem:
            correct_per_class[pred_elem] += 1
            accuracy += 1

    print(f"Ground true labels per class: {true_per_class}")
    print(f"Predicted labels per class: {pred_per_class}")
    print(f"N correct per class: {correct_per_class}")
    print(
        f"Prop true per class: {[true_per_class[n]/len(y_true) for n in [0, 1, 2, 3]]}"
    )
    print(
        f"Prop pred per class: {[pred_per_class[n]/len(y_true) for n in [0, 1, 2, 3]]}"
    )
    print(f"Accuracy: {accuracy/len(y_true):.8f}")
    if y_probs is not None:
        print(f"ROC-AUC : {roc_auc_score(y_true, y_probs, multi_class='ovo'):.6f}")


if __name__ == "__main__":
    exp_n = 88
    path = f"exp_creatinine/exp_{exp_n}/"

    model, test_dl = get_model_and_dl(
        model_dir=path,
        data_path="generated/29var_EH_culled.csv",
        n_layers=4,
        dim_embed=104,
        n_heads=4,
    )

    print("Getting y_probs...")
    y_true, y_probs = get_y_true_y_probs(model, test_dl)

    y_pred = [prob.argmax() for prob in y_probs]

    print_metrics(y_true, y_pred, y_probs)
