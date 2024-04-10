# %%
# Imports and definitions
import os

import numpy as np
import polars as pl

from strats_pytorch.utils import value_to_index

data_dir = "../mimic-iv-2.2/"
output_dir = "generated/"
output_csv_name = "29var_EH.csv"
origins_to_keep = ["EMERGENCY ROOM", "TRANSFER FROM HOSPITAL"]


# %%
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
        return df.filter((pl.col("valuenum") >= low) & (pl.col("valuenum") <= high))

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

    # Blood flow (ml/min) (dialysis)
    df_blood_flow = get_item_lazy(224144)

    # Blood Urea Nitrogen (BUN)
    df_bun = get_item_lazy(225624)

    # Platelet count
    df_plat = get_item_lazy(227457)

    # Lactic Acid
    df_lact = get_item_lazy(225668)

    # O2 saturation pulseoxymetry (SpO2)
    df_spo2 = get_item_lazy(220277)

    # Hemoglobin
    df_hemog = get_item_lazy(220228)

    # Albumin
    df_albu = get_item_lazy(227456)

    # Anion Gap
    df_anion = get_item_lazy(227073)

    # Prothrombin time
    df_prot = get_item_lazy(227465)

    # Arterial O2 Pressure (pO2, arterial)
    df_po2_art = get_item_lazy(220224)

    # Height (cm)
    df_height = get_item_lazy(226730)

    # Glucose (serum)
    df_gluc = get_item_lazy(220621)

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
    df_blood_flow = remove_outside_range(df_blood_flow, 0, 1000)
    df_bun = remove_outside_range(df_bun, 0, 300)
    df_plat = remove_outside_range(df_plat, 0, 500)
    df_lact = remove_outside_range(df_lact, 0, 40)
    df_spo2 = remove_outside_range(df_spo2, 0, 100)
    df_hemog = remove_outside_range(df_hemog, 0, 30)
    df_albu = remove_outside_range(df_albu, 0, 10)
    df_anion = remove_outside_range(df_anion, 0, 50)
    df_prot = remove_outside_range(df_prot, 0, 100)
    df_po2_art = remove_outside_range(df_po2_art, 0, 1000)
    df_height = remove_outside_range(df_height, 0, 300)
    df_gluc = remove_outside_range(df_gluc, 0, 500)

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
            df_blood_flow,
            df_bun,
            df_plat,
            df_lact,
            df_spo2,
            df_hemog,
            df_albu,
            df_anion,
            df_prot,
            df_po2_art,
            df_height,
            df_gluc,
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
        data_dir + "hosp/admissions.csv",
        columns=["admittime", "hadm_id", "deathtime", "admission_location"],
    ).with_columns(pl.col("admittime").str.strptime(pl.Datetime, format="%Y-%m-%d %T"))

    # Add the admission info to the event dataframe
    print("Joining events and admission data...")
    df_ev_hadm = df_ev.join(df_hadm, on="hadm_id")

    # Only keep the stays of patients that come from the specified origins
    print("Removing stays by origin...")
    df_ev_hadm = df_ev_hadm.filter(pl.col("admission_location").is_in(origins_to_keep))

    # Add a column with relative event time (since admission) in minutes
    print("Creating and sorting by 'Time since admission'...")
    df_ev_hadm = df_ev_hadm.with_columns(
        (pl.col("charttime") - pl.col("admittime")).dt.minutes().alias("rel_charttime")
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
    print("Writing to disk...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_ev_hadm_demog.drop(
        ["admission_location", "hadm_id", "stay_id", "subject_id", "admittime"]
    ).write_csv(output_dir + output_csv_name)
    print(f"Done. Wrote {df_ev_hadm_demog.select(pl.count()).item()} lines to csv.")
    print(
        f"Current database contains {len(np.unique(df_ev_hadm_demog.select('ind')))} stays."
    )
