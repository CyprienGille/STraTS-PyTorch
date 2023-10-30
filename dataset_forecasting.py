import numpy as np
import pandas as pd
from typing import Tuple
from torch import Tensor, ones_like, cat
from torch.nn.utils.rnn import pad_sequence

from dataset import MIMIC


class MIMIC_Forecasting(MIMIC):
    def __init__(
        self,
        data_path,
        min_input_len=10,
        target_window_width=60,
        random_past_subset=True,
    ):
        """Load MIMIC-IV data into a Dataset object intended for forecasting


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
        """
        super().__init__(data_path)
        self.min_input_len = min_input_len
        self.target_ww = target_window_width
        self.random_past_subset = random_past_subset

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

    def __getitem__(
        self, index: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        index : int

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
            demog, values, times, variables, fcast_target, fcast_mask
        """
        data = self.df.loc[self.df["ind"] == index].copy()

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
        else:
            # if we want to use the full past except for the last observation
            # (which is reserved for the target)
            input_data = data.iloc[:-1]
            fcast_target, fcast_mask = self._generate_fcast_target(
                data, input_end_idx=-2
            )

        values = Tensor(input_data["valuenum"].to_numpy())
        times = Tensor(input_data["rel_charttime"].to_numpy())
        variables = Tensor(input_data["itemid"].to_numpy())
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
            fcast_target,
            fcast_mask,
        )


# collate_fn for sequences of irregular lengths
def padded_collate_fn(batch: list):
    grouped_inputs = list(zip(*batch))
    (
        demog,
        values,
        times,
        variables,
        fcast_targets,
        fcast_masks,
    ) = grouped_inputs

    padded_values = pad_sequence(values, batch_first=True, padding_value=-1e6)
    padded_times = pad_sequence(times, batch_first=True, padding_value=-1e6)
    padded_variables = pad_sequence(variables, batch_first=True, padding_value=0)

    demog = cat([t.unsqueeze(0) for t in demog])

    mask = ones_like(padded_values)
    mask[padded_values == -1e6] = 0

    fcast_targets = cat([t.unsqueeze(0) for t in fcast_targets])
    fcast_masks = cat([t.unsqueeze(0) for t in fcast_masks])

    return (
        demog,
        padded_values,
        padded_times,
        padded_variables,
        mask,
        fcast_targets,
        fcast_masks,
    )
