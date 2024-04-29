"""Preprocess mimic-iv by selecting features from the top down"""

# Imports and definitions
import json
import os

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from preprocess_mimic_iv import get_item_lazy, remove_outside_range
from strats_pytorch.utils import value_to_index

data_dir = "../mimic-iv-2.2/"
output_dir = "generated/"
n_features = 206  # 206 features stops at 50_000 points per feature
output_csv_name = f"top_{n_features}.csv"
# origins_to_keep = ["EMERGENCY ROOM", "TRANSFER FROM HOSPITAL"]

if __name__ == "__main__":
    # Load chart events selectively, parse dates
    print("Loading chart events... (can take some time)")

    stats = pd.read_csv("generated/mimic_stats.csv", sep=",")
    # Only keep Numeric or Boolean items
    stats = stats[stats["type"] != "Text"]
    df_list = []
    for i in tqdm(range(n_features)):
        df = get_item_lazy(stats.iloc[i]["id"])
        # Remove outliers automatically by using the inter-quantile range
        q1, q3 = df.select(pl.quantile("valuenum", 0.25)), df.select(
            pl.quantile("valuenum", 0.75)
        )
        iqr = q3 - q1
        df = remove_outside_range(df, low=q1 - 3 * iqr, high=q3 + 3 * iqr)
        df_list.append(df)

    df_ev = pl.concat(df_list)

    # reindex the item ids starting from zero
    print("Re-indexing items...")
    new_ids, key = value_to_index(
        df_ev.select("itemid").to_numpy().flatten(),
        cast_from_numpy=True,
        return_key=True,
    )
    df_ev = df_ev.with_columns(
        pl.Series(
            name="itemid",
            values=new_ids,
        )
    )
    with open(output_dir + f"keys_{n_features}.json", "w") as file:
        json.dump(key, file)

    # Load the admissions dataframe, parse dates
    print("Loading admissions...")
    df_hadm = pl.read_csv(
        data_dir + "hosp/admissions.csv",
        columns=["admittime", "hadm_id", "deathtime", "admission_location"],
    ).with_columns(pl.col("admittime").str.strptime(pl.Datetime, format="%Y-%m-%d %T"))

    # Add the admission info to the event dataframe
    print("Joining events and admission data...")
    df_ev_hadm = df_ev.join(df_hadm, on="hadm_id")

    # Only keep the stays of patients that come from the specified origins
    # print("Removing stays by origin...")
    # df_ev_hadm = df_ev_hadm.filter(pl.col("admission_location").is_in(origins_to_keep))

    # Add a column with relative event time (since admission) in minutes
    print("Creating and sorting by 'Time since admission'...")
    df_ev_hadm = df_ev_hadm.with_columns(
        (pl.col("charttime") - pl.col("admittime"))
        .dt.total_minutes()
        .alias("rel_charttime")
    )

    df_ev_hadm = df_ev_hadm.sort([pl.col("hadm_id"), pl.col("rel_charttime")])

    # Add column that indexes the stays starting from zero
    print("Indexing stays...")
    df_ev_hadm = df_ev_hadm.with_columns(
        pl.Series(
            name="ind",
            values=value_to_index(df_ev_hadm.select("stay_id").to_numpy().flatten()),
        )
    )

    # Remove stays with less than 10 measures across all variables
    print("Removing short stays...")
    uniques, counts = np.unique(
        df_ev_hadm.select("ind").to_numpy().flatten(), return_counts=True
    )
    val_counts_dict = dict(zip(uniques, counts))

    df_ev_hadm = df_ev_hadm.with_columns(
        pl.col("ind").replace(val_counts_dict).alias("count")
    )
    df_ev_hadm = df_ev_hadm.filter((pl.col("count") > 9) & (pl.col("count") <= 20000))

    # Add demographic data
    print("Adding demographic data...")
    df_demog = pl.read_csv(
        data_dir + "hosp/patients.csv", columns=["subject_id", "gender", "anchor_age"]
    )

    df_ev_hadm_demog = df_ev_hadm.join(df_demog, on="subject_id")

    print("Writing to disk...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_ev_hadm_demog.drop(
        ["admission_location", "hadm_id", "stay_id", "subject_id", "admittime"]
    ).write_csv(output_dir + output_csv_name)
    print(f"Done. Wrote {df_ev_hadm_demog.select(pl.len()).item()} lines to csv.")
    print(
        f"Current database contains {len(np.unique(df_ev_hadm_demog.select('ind')))} stays."
    )
    print(
        f"Longest stay in current database : {df_ev_hadm_demog.select(pl.col('count').max()).item()} measures"
    )
