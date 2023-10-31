import pandas as pd
from tqdm import tqdm


def creatinine_to_stage(value):
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
val_to_label_func = creatinine_to_stage
data_path = "generated/creat17NoText.csv"
# out_path = data_path
out_path = "generated/creat17Notext_culled.csv"

if __name__ == "__main__":

    print("Reading data...")
    df = pd.read_csv(data_path)

    to_drop_indexes = pd.Index([])

    indexes = df["ind"].unique()

    print("Culling stays...")
    if progress_bar:
        pbar = tqdm(indexes)
    else:
        pbar = indexes

    for ind in pbar:
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

            # Create the label for this stay
            df.loc[df["ind"] == ind, "label"] = val_to_label_func(last_obs_val)
        else:
            # remove stays with no target measure
            to_drop_indexes = to_drop_indexes.append(data.index)

    df.drop(to_drop_indexes, inplace=True)

    print("Writing data...")
    df.to_csv(out_path)
    print(
        f"Done. Current database contains {len(df['ind'].unique())} stays for {len(df.index)} lines."
    )
