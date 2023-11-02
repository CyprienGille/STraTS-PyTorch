#%%
# Imports and definitions
import polars as pl
import numpy as np
import os

data_dir = "../mimic-iv-2.2/"
output_dir = "generated/"
output_csv_name = "creat17NoText.csv"


def value_to_index(vals):
    d = {}
    indexes = []
    free_index = 0  # the lowest unused index
    for id in vals:
        if id not in d.keys():
            # if the id is new
            # allocate to the id the free index
            d[id] = free_index
            free_index += 1
        indexes.append(d[id])
    return indexes


#%%
if __name__ == "__main__":

    def get_item_lazy(itemid: int) -> pl.DataFrame:
        """Get the events for a particular itemid.

        Leverage the speed of polars LazyFrames + batching.
        """
        return (
            pl.scan_csv(data_dir + "icu/chartevents.csv")
            .select(
                [
                    "hadm_id",
                    "stay_id",
                    "subject_id",
                    "charttime",
                    "itemid",
                    "value",
                    "valuenum",
                ]
            )
            .filter(pl.col("itemid").is_in([itemid]))
            .with_columns(
                pl.col("charttime").str.strptime(pl.Datetime, format="%Y-%m-%d %T")
            )
            .collect(streaming=True)
        )

    def remove_outside_range(df, low, high):
        return df.filter((pl.col("valuenum") > low) & (pl.col("valuenum") < high))

    # Load chart events selectively, parse dates
    print("Loading chart events... (can take some time)")

    # Creatinine
    df_creat = get_item_lazy(220615)

    # Heart rate
    df_hr = get_item_lazy(220045)

    # Systolic blood pressure
    df_bp_sys = get_item_lazy(220050)

    # Diastolic blood pressure
    df_bp_dia = get_item_lazy(220051)

    # Mean blood pressure
    df_bp_mean = get_item_lazy(220052)

    # Temperature (Fahrenheit)
    df_temf = get_item_lazy(223761)

    # Weight (Daily/at admission)
    df_weig_D = get_item_lazy(224639)
    df_weig_A = get_item_lazy(226512)

    # WBC
    df_wbc = get_item_lazy(220546)

    # Sodium (serum)
    df_sodium = get_item_lazy(220645)

    # Potassium (serum)
    df_potassium = get_item_lazy(227442)

    # pH (Arterial)
    df_pH = get_item_lazy(223830)

    # Respiratory rate (total)
    df_resp = get_item_lazy(220210)

    # Apnea Interval
    df_apnea = get_item_lazy(223876)

    # Minute volume
    df_vol = get_item_lazy(224687)

    # Central venous pressure
    df_cvp = get_item_lazy(220074)

    # Inspired O2 fraction
    df_insp_o2 = get_item_lazy(223835)

    # Remove outliers
    print("Removing outliers...")
    df_creat = remove_outside_range(df_creat, 0, 10)
    df_hr = remove_outside_range(df_hr, 10, 250)
    df_bp_sys = remove_outside_range(df_bp_sys, 40, 190)
    df_bp_dia = remove_outside_range(df_bp_dia, 30, 120)
    df_bp_mean = remove_outside_range(df_bp_mean, 30, 190)
    df_temf = remove_outside_range(df_temf, 80, 120)
    df_weig_D = remove_outside_range(df_weig_D, 10, 500)
    df_weig_A = remove_outside_range(df_weig_A, 10, 500)
    df_wbc = remove_outside_range(df_wbc, 4, 30)
    df_sodium = remove_outside_range(df_sodium, 100, 200)
    df_potassium = remove_outside_range(df_potassium, 0, 10)
    df_pH = remove_outside_range(df_pH, 7, 8)
    df_resp = remove_outside_range(df_resp, 5, 50)
    df_apnea = remove_outside_range(df_apnea, 5, 75)
    df_vol = remove_outside_range(df_vol, 0.9, 60)
    df_cvp = remove_outside_range(df_cvp, 2, 20)
    df_insp_o2 = remove_outside_range(df_insp_o2, 10, 100)

    print("Concatenating variables...")
    df_ev = pl.concat(
        [
            df_creat,
            df_hr,
            df_bp_sys,
            df_bp_dia,
            df_bp_mean,
            df_temf,
            df_weig_D,
            df_weig_A,
            df_wbc,
            df_sodium,
            df_potassium,
            df_pH,
            df_resp,
            df_apnea,
            df_vol,
            df_cvp,
            df_insp_o2,
        ]
    )

    # reindex the item ids starting from zero
    print("Re-indexing items...")
    df_ev = df_ev.with_columns(
        pl.Series(
            name="itemid",
            values=value_to_index(df_ev.select("itemid").to_numpy().flatten()),
        )
    )

    # Load the admissions dataframe, parse dates
    print("Loading admissions...")
    df_hadm = pl.read_csv(
        data_dir + "hosp/admissions.csv", columns=["admittime", "hadm_id", "deathtime"]
    ).with_columns(pl.col("admittime").str.strptime(pl.Datetime, format="%Y-%m-%d %T"))

    # Add the admission info to the event dataframe
    print("Joining events and admission data...")
    df_ev_hadm = df_ev.join(df_hadm, on="hadm_id")

    # Add a column with relative event time (since admission) in minutes
    print("Creating and sorting by 'Time since admission'...")
    df_ev_hadm = df_ev_hadm.with_columns(
        (pl.col("charttime") - pl.col("admittime")).dt.minutes().alias("rel_charttime")
    )
    # # Group by five minutes
    # df_ev_hadm = df_ev_hadm.with_columns(pl.col("rel_charttime") // 5)

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
        pl.col("ind").map_dict(val_counts_dict).alias("count")
    )
    df_ev_hadm = df_ev_hadm.filter((pl.col("count") > 9) & (pl.col("count") <= 1700))

    # Add demographic data
    print("Adding demographic data...")
    df_demog = pl.read_csv(
        data_dir + "hosp/patients.csv", columns=["subject_id", "gender", "anchor_age"]
    )

    df_ev_hadm_demog = df_ev_hadm.join(df_demog, on="subject_id")

    # Write csv to disk
    # Current n of stays : 70074
    print("Writing to disk...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_ev_hadm_demog.write_csv(output_dir + output_csv_name)
    print(f"Done. Wrote {df_ev_hadm_demog.select(pl.count()).item()} lines to csv.")
    print(
        f"Current database contains {len(np.unique(df_ev_hadm_demog.select('ind')))} stays."
    )
