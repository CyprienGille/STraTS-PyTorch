"""Cull the stays in a preprocessed database to keep only:
- The stays with at least one measure of the target variable
- The part of the stays before (but not including) the last observation of the target variable
Adds a class label for each kept stay according to that last observation.
"""

import pandas as pd
from tqdm import tqdm

from strats_pytorch.utils import value_to_index


def creat_to_4_stages(value):
    """Converts creatinine values to renal risk/injury/failure stages
    according to the KDIGO criteria

    Parameters
    ----------
    value : float
        Creatinine (serum) value, in mg/dL

    Returns
    -------
    int
        0: Normal; 1: Risk; 2: Injury; 3: Failure
    """
    if value < 1.35:
        return 0
    elif value < 2.68:
        return 1
    elif value < 4.16:
        return 2
    return 3


# Params
progress_bar = True
tgt_item_id = 0  # the id of the target variable
keep_tgt_var = False
val_to_label_func = creat_to_4_stages
data_path = "generated/29var_EH.csv"
out_path = "generated/29var_EH_culled.csv"


if __name__ == "__main__":
    print("Reading data...")
    df = pd.read_csv(data_path)

    to_drop_indexes = pd.Index([])

    ind_to_label = {}

    indexes = df["ind"].unique()

    print("Culling stays...")
    if progress_bar:
        pbar = tqdm(indexes)
    else:
        pbar = indexes

    for ind in pbar:
        # Note: this can in theory be parallelized
        data = df[df["ind"] == ind].copy()
        try:
            # get the index of the last target observation
            last_obs_idx = data[data["itemid"] == tgt_item_id].index[-1]
            idx_ok = True
        except IndexError:  # No target measure in the stay
            idx_ok = False

        if idx_ok:
            last_obs_time = data.loc[last_obs_idx, "rel_charttime"]
            last_obs_val = data.loc[last_obs_idx, "valuenum"]

            # Only keep the part of the stay before the last observation of the target variable
            to_drop = data[data["rel_charttime"] >= last_obs_time]
            to_drop_indexes = to_drop_indexes.append(to_drop.index)
            if not keep_tgt_var:
                to_drop_indexes = to_drop_indexes.append(
                    data[data["itemid"] == tgt_item_id].index
                )

            # Create the label for this stay
            ind_to_label[ind] = val_to_label_func(last_obs_val)
        else:
            # remove stays with no target measure
            to_drop_indexes = to_drop_indexes.append(data.index)

    df.drop(to_drop_indexes, inplace=True)
    # add labels
    indexes = df["ind"].unique()
    if progress_bar:
        pbar = tqdm(indexes)
    else:
        pbar = indexes
    for ind in pbar:
        df.loc[df["ind"] == ind, "label"] = ind_to_label[ind]

    # reindex the stays
    df["ind"] = value_to_index(df["ind"])

    print("Writing data...")
    df.to_csv(out_path)
    print(
        f"Done. Current database contains {len(df['ind'].unique())} stays for {len(df.index)} lines."
    )
