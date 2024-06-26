"""A One-to-Many approach to embedding values"""

from torch import Tensor, nn


class CVE(nn.Module):
    def __init__(self, hid_units, output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim

        self.lin_1 = nn.Linear(in_features=1, out_features=hid_units)
        self.lin_2 = nn.Linear(in_features=hid_units, out_features=output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor):
        """
        [batch_size, seq_len] -> [batch_size, seq_len, output_dim]
        """

        return self.lin_2(self.tanh(self.lin_1(x.unsqueeze(-1))))
