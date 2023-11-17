import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataset_int_classif import MIMIC_Int_Classif, padded_collate_fn
from models.strats import STraTS


def get_y_true_y_probs(
    model: STraTS,
    test_dl: DataLoader,
):
    y_true = []
    y_probs = []
    # Testing mode
    model.eval()

    for demog, values, times, variables, tgt, masks in tqdm(test_dl):
        pred = model(demog, values, times, variables, masks)

        y_true.append(tgt.item())
        y_probs.append(pred.detach())
    return y_true, y_probs


def print_metrics(y_true, y_pred, y_probs):
    # TODO Add other metrics than just acc

    true_per_class = {0: 0, 1: 0, 2: 0, 3: 0}
    pred_per_class = {0: 0, 1: 0, 2: 0, 3: 0}
    accuracy = 0
    for i, elem in enumerate(y_true):
        true_per_class[elem] += 1

        pred_elem = y_pred[i]
        pred_per_class[pred_elem] += 1

        if elem == pred_elem:
            accuracy += 1

    print(f"Ground true labels per class: {true_per_class}")
    print(f"Predicted labels per class: {pred_per_class}")
    print(
        f"Prop true per class: {[true_per_class[n]/len(y_true) for n in [0, 1, 2, 3]]}"
    )
    print(
        f"Prop pred per class: {[pred_per_class[n]/len(y_true) for n in [0, 1, 2, 3]]}"
    )
    print(f"Accuracy: {accuracy/len(y_true)}")


if __name__ == "__main__":
    exp_n = 0
    path = f"exp_creatinine/exp_{exp_n}/"

    test_ds = MIMIC_Int_Classif(
        data_path="generated/creat17NoText_culled.csv",
    )
    test_ds.restrict_to_indexes(np.load(path + "test_idx.npy"))
    test_ds.normalize(normalize_times=True, normalize_vars=True, verbose=True)

    test_dl = DataLoader(
        test_ds, batch_size=1, collate_fn=padded_collate_fn, shuffle=False
    )

    model = STraTS(
        n_var_embs=17,
        dim_demog=2,
        dropout=0.0,
        n_layers=2,
        dim_embed=52,
        n_heads=4,
        forecasting=False,
        n_classes=4,
    )
    model.load_state_dict(torch.load(path + "STraTS.pth"))

    print("Getting y_probs...")
    y_true, y_probs = get_y_true_y_probs(model, test_dl)

    y_pred = [prob.argmax().item() for prob in y_probs]

    print_metrics(y_true, y_pred, y_probs)
