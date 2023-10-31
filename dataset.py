import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from preprocess_mimic_iv import value_to_index


class MIMIC(Dataset):
    def __init__(
        self,
        data_path,
    ):
        """Load MIMIC-IV data into a Dataset object


        Parameters
        ----------
        data_path : str
            Path to a .csv containing data (check preprocessing code to see what that data looks like)
        """
        super().__init__()

        self.df = pd.read_csv(
            data_path,
            parse_dates=True,
        )

        self.indexes = self.df["ind"].unique()

    def restrict_to_indexes(self, indexes):
        self.indexes = indexes
        self.df = self.df.loc[self.df["ind"].isin(indexes)]

    def normalize(self, normalize_vars=True, normalize_times=True):
        """Note: per-stay time normalization can be slow if there are a lot of stays."""
        if normalize_vars:
            # per-variable normalization
            self.means = {}
            self.stds = {}
            for unique_itemid in self.df["itemid"].unique():
                data = self.df.loc[self.df["itemid"] == unique_itemid][
                    "valuenum"
                ].copy()
                self.means[unique_itemid] = data.mean()
                self.stds[unique_itemid] = data.std(ddof=0)
                self.df.loc[self.df["itemid"] == unique_itemid, "valuenum"] = (
                    data - data.mean()
                ) / data.std(ddof=0)

            # age normalization
            data = self.df["anchor_age"].copy()
            self.age_mean = data.mean()
            self.age_std = data.std(ddof=0)
            self.df["anchor_age"] = (data - self.age_mean) / self.age_std

        if normalize_times:
            # per-stay time normalization
            # Note: can be time consuming
            self.time_means = {}
            self.time_stds = {}
            for ind in self.df["ind"].unique():
                data = self.df.loc[self.df["ind"] == ind]["rel_charttime"].copy()
                self.time_means[ind] = data.mean()
                self.time_stds[ind] = data.std(ddof=0)
                self.df.loc[self.df["ind"] == ind, "rel_charttime"] = (
                    data - data.mean()
                ) / data.std(ddof=0)

    def __len__(self) -> int:
        """Number of stays in this dataset."""
        return len(self.indexes)
