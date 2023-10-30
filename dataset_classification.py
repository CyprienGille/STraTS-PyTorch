import numpy as np
from typing import Tuple
from torch import Tensor, ones_like, cat
from torch.utils.data import WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence

from dataset import MIMIC


class MIMIC_Classification(MIMIC):
    def __init__(
        self,
        data_path,
        min_input_len=10,
        random_past_subset=True,
    ):
        """Load MIMIC-IV data into a Dataset object intended for external classification
        (Example: mortality prediction)


        Parameters
        ----------
        data_path : str
            Path to a .csv containing data (check preprocessing code to see what that data looks like)
        min_input_len : int, optional
            The minimal width of any observation window, in number of measures, by default 10
        random_past_subset : bool, optional
            Whether to use a random observation window, by default True
        """
        super().__init__(data_path)

        self.min_input_len = min_input_len
        self.random_past_subset = random_past_subset

    def get_sampler(self, weight_class_0=1, weight_class_1=1, replacement=True):
        """Generates a WeightedRandomSampler with class-dependent weights.
        Those weights do not have to sum to 1.

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

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        index : int

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            demog, values, times, variables, died
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
        else:
            input_data = data

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
        )


# collate_fn for sequences of irregular lengths
def padded_collate_fn(batch: list):
    grouped_inputs = list(zip(*batch))
    (
        demog,
        values,
        times,
        variables,
        died,
    ) = grouped_inputs

    padded_values = pad_sequence(values, batch_first=True, padding_value=-1e6)
    padded_times = pad_sequence(times, batch_first=True, padding_value=-1e6)
    padded_variables = pad_sequence(variables, batch_first=True, padding_value=0)

    demog = cat([t.unsqueeze(0) for t in demog])

    died = Tensor(died)

    mask = ones_like(padded_values)
    mask[padded_values == -1e6] = 0

    return (
        demog,
        padded_values,
        padded_times,
        padded_variables,
        died,
        mask,
    )
