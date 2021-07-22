from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor


class LatticePositionalEmbedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, tied_embs=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.tied_embs = tied_embs
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings

        self.topo_from = self.new_emb()

        if tied_embs:
            self.topo_to = self.topo_from
        else:
            self.topo_to = self.new_emb()

    def new_emb(self):
        m = nn.Embedding(self.num_embeddings, self.embedding_dim, padding_idx=self.padding_idx)
        nn.init.normal_(m.weight, mean=0, std=self.embedding_dim ** -0.5)
        if self.padding_idx is not None:
            nn.init.constant_(m.weight[self.padding_idx], 0)
        return m

    def forward(
        self,
        input: Tensor,
        positions: Optional[Tensor] = None,):
    # incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,):
        # input - tensor btz X sl
        # positions - tensor btz X sl X (2) . [[[, state_from, state_to] , ... ], ...]
        emb_from = self.topo_from(positions[:, :, -2])
        emb_to = self.topo_to(positions[:, :, -1])
        return (emb_from + emb_to)/2
        #return emb_from + emb_to
