"""Pytorch Implementation of STraTS (Tipirneni & Reddy, 2022)"""
import torch
from torch import nn
from math import sqrt
from typing import Optional

from models.cve import CVE
from models.attention import Attention
from models.transformer import TransformerEncoderWrapper


class STraTS(nn.Module):

    def __init__(self, 
                n_var_embs:int, 
                dim_demog:int=2, 
                dim_embed:int=52, 
                n_layers:int=2, 
                n_heads:int=4, 
                dim_ff:int=None, 
                dropout:float=0.0,
                forecasting:bool=False,
                ) -> None:
        super().__init__()

        self.lin_1_demog = nn.Linear(in_features=dim_demog, out_features=2*dim_embed)
        self.lin_2_demog = nn.Linear(in_features=2*dim_embed, out_features=dim_embed)
        self.tanh_demog = nn.Tanh()
        self.var_embedding = nn.Embedding(num_embeddings=n_var_embs, embedding_dim=dim_embed)
        self.cve_values = CVE(hid_units=int(sqrt(dim_embed)), output_dim=dim_embed)
        self.cve_times = CVE(hid_units=int(sqrt(dim_embed)), output_dim=dim_embed)
        self.trans = TransformerEncoderWrapper(dim_embed, n_layers, n_heads, dim_ff, dropout)
        self.attn = Attention(dim_embed, hid_dim=2*dim_embed)

        self.forecasting = forecasting

        self.ouput_forecasting = nn.Linear(in_features=2*dim_embed, out_features=n_var_embs)
        self.ouput_prediction = nn.Linear(in_features=2*dim_embed, out_features=2)
        self.softmax_prediction = nn.Softmax(dim=-1)


    def forward(self, demog:torch.Tensor, variables:torch.Tensor, values:torch.Tensor, times:torch.Tensor, mask:Optional[torch.Tensor]=None):
        # This just follows figure 3 of the STraTS paper
        # Note: the mask is for unobserved variables

        demo_emb = self.lin_2_demog(self.tanh_demog(self.lin_1_demog(demog)))
        var_emb = self.var_embedding(variables.long())
        values_emb = self.cve_values(values)
        times_emb = self.cve_times(times)
        combined_emb = var_emb + values_emb + times_emb # TODO: Better way of combining the embeddings?
        contextual_emb = self.trans(combined_emb, src_key_padding_mask=mask)
        attn_weights = self.attn(contextual_emb, mask=mask)
        fused_emb = torch.sum(contextual_emb*attn_weights, dim=-2)
        concat = torch.concat((fused_emb, demo_emb), dim=1)
        if self.forecasting:
            return self.ouput_forecasting(concat)
        return self.softmax_prediction(self.ouput_prediction(concat).squeeze())

class STraTS_no_demog(nn.Module):

    def __init__(self, 
                n_var_embs:int, 
                dim_embed:int=52, 
                n_layers:int=2, 
                n_heads:int=4, 
                dim_ff:int=None, 
                dropout:float=0.0,
                forecasting:bool=False,
                ) -> None:
        super().__init__()

        self.var_embedding = nn.Embedding(num_embeddings=n_var_embs, embedding_dim=dim_embed)
        self.cve_values = CVE(hid_units=int(sqrt(dim_embed)), output_dim=dim_embed)
        self.cve_times = CVE(hid_units=int(sqrt(dim_embed)), output_dim=dim_embed)
        self.trans = TransformerEncoderWrapper(dim_embed, n_layers, n_heads, dim_ff, dropout)
        self.attn = Attention(dim_embed, hid_dim=2*dim_embed)

        self.forecasting = forecasting

        self.ouput_forecasting = nn.Linear(in_features=dim_embed, out_features=n_var_embs)
        self.ouput_prediction = nn.Linear(in_features=dim_embed, out_features=2)
        self.softmax_prediction = nn.Softmax(dim=-1)


    def forward(self, variables:torch.Tensor, values:torch.Tensor, times:torch.Tensor, mask:Optional[torch.Tensor]=None):        
        
        var_emb = self.var_embedding(variables.long())
        values_emb = self.cve_values(values)
        times_emb = self.cve_times(times)
        combined_emb = var_emb + values_emb + times_emb
        contextual_emb = self.trans(combined_emb, src_key_padding_mask=mask)
        attn_weights = self.attn(contextual_emb, mask=mask)
        fused_emb = torch.sum(contextual_emb*attn_weights, dim=-2)
        if self.forecasting:
            return self.ouput_forecasting(fused_emb)
        return self.softmax_prediction(self.ouput_prediction(fused_emb).squeeze())

        