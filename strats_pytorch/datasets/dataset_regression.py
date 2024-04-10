from typing import Optional, Tuple

from torch import Tensor, cat, ones_like
from torch.nn.utils.rnn import pad_sequence

from strats_pytorch.datasets.dataset import MIMIC
from strats_pytorch.utils import denorm, norm


class MIMIC_Reg(MIMIC):
    def __init__(
        self,
        data_path: str,
        var_id: int = 0,
        crop_back_interval: Optional[int] = None,
        keep_tgt_var: bool = False,
    ):
        """Load MIMIC-IV data into a Dataset object intended for regression

        Parameters
        ----------
        data_path : str
            path to the preprocessed csv
        var_id : int, optional
            id of the target variable, by default 0
        crop_back_interval : int, optional
            If not None, how many minutes before the target time to keep, by default None
        keep_tgt_var : bool, optional
            whether to keep the target variable in the input (should not be set to True for any training purposes), by default False
        """
        super().__init__(data_path)
        self.var_id = var_id
        self.back = crop_back_interval
        self.keep_tgt_var = keep_tgt_var

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
            demog, values, times, variables, target_value, target_time
        """

        data = self.df.loc[self.df["ind"] == self.indexes[index]].copy()

        try:
            tgt_line = data.loc[data["itemid"] == self.var_id].iloc[-1]
            tgt_val = Tensor([tgt_line["valuenum"]])
            tgt_time = Tensor([tgt_line["rel_charttime"]])
        except IndexError:
            raise IndexError(
                "This stay has no measure for the target variable and thus cannot be used for regression. Run culling_reg.py to remove such stays."
            )

        if not self.keep_tgt_var:
            data = data.loc[data["itemid"] != self.var_id]

        values = Tensor(data["valuenum"].to_numpy())
        times = Tensor(data["rel_charttime"].to_numpy())
        variables = Tensor(data["itemid"].to_numpy())

        if self.back is not None:
            # Get the indices of times that are at most [back] minutes before tgt time
            cutoff_time = (
                tgt_time.item() - self.back
                if not self.normed_times
                else norm(
                    denorm(tgt_time.item(), self.time_mean, self.time_std) - self.back,
                    self.time_mean,
                    self.time_std,
                )
            )
            to_keep = times >= cutoff_time
            values = values[to_keep]
            times = times[to_keep]
            variables = variables[to_keep]

        gender = Tensor([-1.0]) if data["gender"].iloc[0] == "M" else Tensor([1.0])
        age = Tensor([data["anchor_age"].iloc[0]])
        demog = cat([gender, age])

        return (
            demog,
            values,
            times,
            variables,
            tgt_val,
            tgt_time,
        )


# collate_fn for sequences of irregular lengths
def padded_collate_fn(batch: list):
    grouped_inputs = list(zip(*batch))
    (demog, values, times, variables, tgt_vals, tgt_times) = grouped_inputs

    padded_values = pad_sequence(values, batch_first=True, padding_value=-1e4)
    padded_times = pad_sequence(times, batch_first=True, padding_value=-1e4)
    padded_variables = pad_sequence(variables, batch_first=True, padding_value=47)

    demog = cat([t.unsqueeze(0) for t in demog])

    tgt_vals = Tensor(tgt_vals)
    tgt_times = Tensor(tgt_times)

    masks = ones_like(padded_values)
    masks[padded_values == -1e4] = 0

    return (
        demog,
        padded_values,
        padded_times,
        padded_variables,
        tgt_vals,
        tgt_times,
        masks,
    )
