import os

import pandas as pd
from tqdm import tqdm

from strats_pytorch.utils import value_to_index

# Params
split_into_files = True  # create individual stay files
progress_bar = True
tgt_item_id = 47  # the id of the target variable
will_keep_tgt_var = False
data_path = "generated/top_206.csv"
out_path = "generated/top_206_culled_reg.csv"


if __name__ == "__main__":
    print("Reading data...")
    df = pd.read_csv(data_path)

    to_drop_indexes = pd.Index([], dtype=int)

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
            to_drop = data[data["rel_charttime"] > last_obs_time]
            to_drop_indexes = to_drop_indexes.append(to_drop.index)

            if not will_keep_tgt_var:
                # Remove stays that only have measures for the target variable
                # and that are thus unusable if we don't keep it during training
                non_tgt_in_stay = data[
                    (data["rel_charttime"] <= last_obs_time)
                    & (data["itemid"] != tgt_item_id)
                ]
                if len(non_tgt_in_stay) == 0:
                    to_drop_indexes = to_drop_indexes.append(
                        data[data["rel_charttime"] <= last_obs_time].index
                    )

        else:
            # remove stays with no target measure
            to_drop_indexes = to_drop_indexes.append(data.index)

    df.drop(to_drop_indexes, inplace=True)
    # reindex the stays
    df["ind"] = value_to_index(df["ind"])

    ## Bottom-up (29 var)
    # Latest database contains 66324 stays for 19 461 981 lines
    # Latest database (EH) contains 46369 stays for 13 552 031 lines
    ## Top Down (206 var)
    # Latest database contains 69023 stays for 75 802 088 lines

    print("Writing data...")
    if split_into_files:
        out_dir = out_path[:-4] + "/"
        os.makedirs(out_dir, exist_ok=True)

        for ind in tqdm(df["ind"].unique()):
            df[df["ind"] == ind].to_csv(out_dir + f"ind_{ind}.csv")

    df.to_csv(out_path)
    print(
        f"Done. Current database contains {len(df['ind'].unique())} stays for {len(df.index)} lines."
    )
