from typing import Tuple
from torch import Tensor, ones_like, cat
from torch.nn.utils.rnn import pad_sequence

from dataset import MIMIC


class MIMIC_Int_Classif(MIMIC):
    def __init__(self, data_path):
        """Load MIMIC-IV data into a Dataset object intended for internal classification

        Parameters
        ----------
        data_path : str
            Path to the .csv containing preprocessed data
        """
        super().__init__(data_path)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        index : int

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            demog, values, times, variables, target
        """

        data = self.df.loc[self.df["ind"] == self.indexes[index]].copy()

        values = Tensor(data["valuenum"].to_numpy())
        times = Tensor(data["rel_charttime"].to_numpy())
        variables = Tensor(data["itemid"].to_numpy())

        try:
            tgt = Tensor([data["label"].to_numpy()[0]])
        except KeyError:
            raise KeyError(
                "'label'. You might have forgotten to run culling.py on this database."
            )

        gender = Tensor([-1.0]) if data["gender"].iloc[0] == "M" else Tensor([1.0])
        age = Tensor([data["anchor_age"].iloc[0]])
        demog = cat([gender, age])

        return (
            demog,
            values,
            times,
            variables,
            tgt,
        )


# collate_fn for sequences of irregular lengths
def padded_collate_fn(batch: list):
    grouped_inputs = list(zip(*batch))
    (
        demog,
        values,
        times,
        variables,
        tgt,
    ) = grouped_inputs

    padded_values = pad_sequence(values, batch_first=True, padding_value=-1e6)
    padded_times = pad_sequence(times, batch_first=True, padding_value=-1e6)
    padded_variables = pad_sequence(variables, batch_first=True, padding_value=0)

    demog = cat([t.unsqueeze(0) for t in demog])

    tgt = Tensor(tgt)

    mask = ones_like(padded_values)
    mask[padded_values == -1e6] = 0

    return (
        demog,
        padded_values,
        padded_times,
        padded_variables,
        tgt,
        mask,
    )
