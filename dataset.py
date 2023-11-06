import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset


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

    def normalize(self, normalize_vars=True, normalize_times=True, verbose=False):
        """Note: per-stay time normalization can be slow if there are a lot of stays.
        TODO Try global time normalization
        """
        if normalize_vars:
            if verbose:
                print("Normalizing variables...")
                itemids = tqdm(self.df["itemid"].unique())
            else:
                itemids = self.df["itemid"].unique()
            # per-variable normalization
            self.means = {}
            self.stds = {}
            for unique_itemid in itemids:
                data = self.df.loc[
                    self.df["itemid"] == unique_itemid, "valuenum"
                ].copy()

                mean = data.mean()
                std = data.std(ddof=0)

                self.means[unique_itemid] = mean
                self.stds[unique_itemid] = std

                if std == 0:
                    # if there is only one value for this itemid
                    self.df.loc[self.df["itemid"] == unique_itemid, "valuenum"] = (
                        data - mean
                    )
                else:
                    self.df.loc[self.df["itemid"] == unique_itemid, "valuenum"] = (
                        data - mean
                    ) / std

            # age normalization
            data = self.df["anchor_age"].copy()
            self.age_mean = data.mean()
            self.age_std = data.std(ddof=0)
            self.df["anchor_age"] = (data - self.age_mean) / self.age_std

        if normalize_times:
            if verbose:
                print("Normalizing times...")
                inds = tqdm(self.df["ind"].unique())
            else:
                inds = self.df["ind"].unique()
            # per-stay time normalization
            # Note: can be time consuming
            self.time_means = {}
            self.time_stds = {}
            for ind in inds:
                data = self.df.loc[self.df["ind"] == ind, "rel_charttime"].copy()

                mean = data.mean()
                std = data.std(ddof=0)

                self.time_means[ind] = mean
                self.time_stds[ind] = std

                if std == 0:
                    # If there is only one time value
                    self.df.loc[self.df["ind"] == ind, "rel_charttime"] = data - mean
                else:
                    self.df.loc[self.df["ind"] == ind, "rel_charttime"] = (
                        data - mean
                    ) / std

    def __len__(self) -> int:
        """Number of stays in this dataset."""
        return len(self.indexes)
