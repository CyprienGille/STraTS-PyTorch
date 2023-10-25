import numpy as np
import pandas as pd
from typing import Tuple, Optional
from torch import Tensor, ones_like, cat
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence

from preprocess_mimic_iv import value_to_index


class MIMIC_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        min_input_len=10,
        target_window_width=60,
        random_past_subset=True,
        classif_target_name: Optional[str] = "creat_label",
    ):
        """Load MIMIC-IV data into a Dataset object

        Important note: Do not set random_past_subset to false unless you're doing
        death classification/another **external** supervision task.
        For example, for creatinine classification, if random_past_subset=False,
        the only supervised stays will be those where the last measure taken is a
        Creatinine (serum) measure!

        Parameters
        ----------
        data_path : str
            Path to a .csv containing data (check preprocessing code to see what that data looks like)
        min_input_len : int, optional
            The minimal width of any observation window, in number of measures, by default 10
        target_window_width : int, optional
            The number of minutes to look ahead for a target observation, by default 60
        random_past_subset : bool, optional
            Whether to use a random observation window, by default True
        classif_target_name : Optional[str], optional
            The name of the label column in the data, by default "creat_label". If None, placeholder tensors will be returned by getitem in place of the internal classification target and mask.
        """
        super().__init__()

        self.min_input_len = min_input_len
        self.target_ww = target_window_width
        self.random_past_subset = random_past_subset
        self.classif_target_name = classif_target_name
        self.sampler = None

        self.df = pd.read_csv(
            data_path,
            parse_dates=True,
            usecols=[
                "ind",
                "rel_charttime",
                "valuenum",
                "itemid",
                "deathtime",
                "count",
                "gender",
                "anchor_age",
                self.classif_target_name,
            ],
        )

    def restrict_to_indexes(self, indexes):
        self.df = self.df.loc[self.df["ind"].isin(indexes)]
        self.reindex()

    def reindex(self):
        self.df["ind"] = value_to_index(self.df["ind"])

    def normalize(self, normalize_vars=True, normalize_time=True):
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

        if normalize_time:
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

    def get_sampler(self, weight_class_0=1, weight_class_1=1, replacement=True):
        """Generates a WeightedRandomSampler with class-dependent weights.
        Those weights do not have to sum to 1.

        Note: Only works for the mortality classification task for now.

        Parameters
        ----------
        weight_class_0 : int or float, optional
            The weight of samples of class 0 (lived), by default 1
        weight_class_1 : int or float, optional
            The weight of samples of class 1 (died), by default 1
        replacement : bool, optional
            Whether samples should be drawn with replacement from the sampler, by default True

        Returns
        -------
        torch.utils.data.WeightedRandomSampler
            The random sampler with per-sample weights according to the labels
        """
        if (
            self.sampler is not None
            and self.w0 == weight_class_0
            and self.w1 == weight_class_1
        ):
            return self.sampler  # avoid re-generating the sampler

        self.w0 = weight_class_0
        self.w1 = weight_class_1

        weights = []

        if self.w0 == self.w1:
            # No need to check the labels
            weights = [self.w0] * self.__len__()
        else:
            for ind in range(self.__len__()):
                data = self.df.loc[self.df["ind"] == ind].copy()
                survived = data["deathtime"].hasnans
                if survived:
                    weights.append(self.w0)
                else:
                    weights.append(self.w1)

        self.sampler = WeightedRandomSampler(
            weights=weights, num_samples=self.__len__(), replacement=replacement
        )
        return self.sampler

    def _generate_fcast_target(
        self, data: pd.DataFrame, input_end_idx: int
    ) -> Tuple[Tensor, Tensor]:
        target = []
        mask = []
        input_end_time = data.iloc[input_end_idx]["rel_charttime"]

        for unique_itemid in self.df["itemid"].unique():

            item_target = data.loc[
                (data["rel_charttime"] > input_end_time)
                & (data["rel_charttime"] <= input_end_time + self.target_ww)
                & (data["itemid"] == unique_itemid)
            ]
            if len(item_target) > 0:
                # Note: if there are several observations in the obs window,
                # we take the 1st one (TODO test with the mean?)
                target.append(item_target["valuenum"].iloc[0])
                mask.append(1.0)
            else:
                target.append(0.0)
                mask.append(0.0)
        return Tensor(target), Tensor(mask)

    def _generate_classif_target(
        self, data: pd.DataFrame, input_end_idx, ignore_index=-1
    ) -> Tensor:
        target = []
        input_end_time = data.iloc[input_end_idx]["rel_charttime"]

        future_data = data.loc[
            (data["rel_charttime"] > input_end_time)
            & (data["rel_charttime"] <= input_end_time + self.target_ww)
            & (data[self.classif_target_name] != -1)
        ]
        if len(future_data) > 0:
            # Note: if there are several observations in the obs window,
            # we only take the 1st one
            target = [future_data[self.classif_target_name].iloc[0]]
        else:
            target = [ignore_index]  # Index ignored in the loss function later
        return Tensor(target)

    def __getitem__(
        self, index: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        index : int

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
            demog, values, times, variables, died, fcast_target, fcast_mask, classif_target
        """
        data = self.df.loc[self.df["ind"] == index].copy()
        classif_target = Tensor([-1.0])

        if self.random_past_subset:
            # if we want to select a random subset of the observations as the input data
            count = data["count"].iloc[0]  # number of observations for this ICU stay

            if self.min_input_len > count - 1:
                input_len = self.min_input_len
            else:
                input_len = np.random.randint(self.min_input_len, count - 1)

            input_start_idx = np.random.randint(0, count - input_len)
            input_data = data.iloc[input_start_idx : input_start_idx + input_len]
            fcast_target, fcast_mask = self._generate_fcast_target(
                data, input_end_idx=input_start_idx + input_len - 1
            )
            if self.classif_target_name is not None:
                classif_target = self._generate_classif_target(
                    data, input_end_idx=input_start_idx + input_len - 1
                )
        else:
            # if we want to use the full past except for the last observation
            # (which is reserved for the target)
            input_data = data.iloc[:-1]
            fcast_target, fcast_mask = self._generate_fcast_target(
                data, input_end_idx=-2
            )
            if self.classif_target_name is not None:
                classif_target = self._generate_classif_target(data, input_end_idx=-2)

        values = Tensor(input_data["valuenum"].to_numpy())
        times = Tensor(input_data["rel_charttime"].to_numpy())
        variables = Tensor(input_data["itemid"].to_numpy())
        died = Tensor([0.0]) if input_data["deathtime"].hasnans else Tensor([1.0])
        gender = (
            Tensor([-1.0]) if input_data["gender"].iloc[0] == "M" else Tensor([1.0])
        )
        age = Tensor([input_data["anchor_age"].iloc[0]])
        demog = cat([gender, age])

        return (
            demog,
            values,
            times,
            variables,
            died,
            fcast_target,
            fcast_mask,
            classif_target,
        )

    def __len__(self) -> int:
        """Number of stays in this dataset."""
        return len(self.df["ind"].unique())


# collate_fn for sequences of irregular lengths
def padded_collate_fn(batch: list):
    grouped_inputs = list(zip(*batch))
    (
        demog,
        values,
        times,
        variables,
        died,
        fcast_targets,
        fcast_masks,
        classif_targets,
    ) = grouped_inputs

    padded_values = pad_sequence(values, batch_first=True, padding_value=-1e6)
    padded_times = pad_sequence(times, batch_first=True, padding_value=-1e6)
    padded_variables = pad_sequence(variables, batch_first=True, padding_value=0)

    demog = cat([t.unsqueeze(0) for t in demog])

    died = Tensor(died)

    mask = ones_like(padded_values)
    mask[padded_values == -1e6] = 0

    # ensure that everything has the right shape

    fcast_targets = cat([t.unsqueeze(0) for t in fcast_targets])
    fcast_masks = cat([t.unsqueeze(0) for t in fcast_masks])

    classif_targets = Tensor(classif_targets)

    return (
        demog,
        padded_values,
        padded_times,
        padded_variables,
        died,
        mask,
        fcast_targets,
        fcast_masks,
        classif_targets,
    )
