from typing import Optional

from torch import nn


class TransformerEncoderWrapper(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_ff: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.d_model, self.n_layers, self.n_heads, self.dropout, self.activ = (
            d_model,
            n_layers,
            n_heads,
            dropout,
            activation,
        )
        self.dim_ff = dim_ff or 2 * d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.dim_ff,
            dropout=self.dropout,
            batch_first=True,
            activation=self.activ,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )

    def forward(self, src, mask=None, src_key_padding_mask=None):
        return self.transformer(
            src, mask=mask, src_key_padding_mask=src_key_padding_mask
        )
