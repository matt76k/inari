from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import BoolTensor, Tensor
from torch_geometric.utils import scatter


class GlobalAttentionPooling(nn.Module):
    def __init__(self, gate_nn, feat_nn=None):
        super().__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn

    def forward(self, x, batch, size: Optional[int] = None) -> Tensor:
        gate = F.softmax(self.gate_nn(x), dim=1)
        feat = self.feat_nn(x) if self.feat_nn else x

        r = feat * gate

        dim = -1 if x.dim() == 1 else -2
        if batch is None:
            return r.sum(dim=dim, keepdim=x.dim() <= 2)[0]
        size = int(batch.max().item() + 1) if size is None else size
        return scatter(r, batch, dim=dim, dim_size=size, reduce="sum")


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        self.heads = heads
        self.dim = dim

        super().__init__()
        self.Q = torch.nn.Parameter(torch.Tensor(self.heads, self.dim, self.dim))
        self.K = torch.nn.Parameter(torch.Tensor(self.heads, self.dim, self.dim))
        self.V = torch.nn.Parameter(torch.Tensor(self.heads, self.dim, self.dim))
        self.last = torch.nn.Parameter(torch.Tensor(self.heads * self.dim, self.dim))
        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.last)

    def forward(self, q, k, v, mask):
        q = torch.matmul(q, self.Q)
        k = torch.matmul(k, self.K)
        v = torch.matmul(v, self.V)

        att = torch.matmul(k, torch.permute(q, (0, 2, 1)))
        fmask = mask.to(torch.float32)
        dim = q.size(1)
        att *= fmask
        att = att / (dim**0.5)
        att += -1e9 * (~mask).to(torch.float32)
        att = F.softmax(att, dim=2)

        att = rearrange(torch.matmul(att, v), "heads n d -> n (heads d)")

        return torch.mm(att, self.last)


class SimpleMM(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, embed_t: Tensor, embed_q: Tensor, mask: BoolTensor) -> Tensor:
        fmask = mask.to(torch.float32)

        dim = embed_t.size(1)
        att = torch.mm(embed_q, embed_t.T)
        att *= fmask
        att = att / (dim**0.5)
        att += -1e9 * (~mask).to(torch.float32)
        att = F.softmax(att, dim=1)

        return att


class AffinityMM(nn.Module):
    # Learning Combinatorial Embedding Networks for Deep Graph Matching
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.W = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.tau = nn.Parameter(torch.empty((1, 1)))

        nn.init.xavier_uniform_(self.W)
        nn.init.constant_(self.tau, 1.0)

    def forward(self, embed_t, embed_q, mask):
        att = torch.mm(embed_t, self.W)
        att = torch.mm(embed_q, att.T)

        fmask = mask.to(torch.float32)

        att *= fmask
        att = att / self.tau
        att += -1e9 * (~mask).to(torch.float32)
        att = F.softmax(att, dim=1)

        return att


class NTN(nn.Module):
    def __init__(self, dim, k):
        self.k = k
        self.dim = dim

        super().__init__()
        self.W = torch.nn.Parameter(torch.Tensor(self.k, self.dim, self.dim))
        self.bias = torch.nn.Parameter(torch.empty((self.k, 1, 1)))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.bias)

    def forward(self, embed_t, embed_q):
        score = torch.mm(embed_t, rearrange(self.W, "k h w -> h (k w)"))
        score = rearrange(score, "n (k d) -> k d n", k=self.k)
        score = torch.matmul(rearrange(embed_q, "n d -> 1 n d"), score)

        #        q = repeat(embed_q, 'n d -> (n m) d', m=embed_t.size(0))
        #        t = repeat(embed_t, 'n d -> (n m) d', m=embed_q.size(0))
        #        st = torch.cat([q, t], dim=1)
        #        st = torch.mm(self.V, st.T)
        #        st = rearrange(st, 'k (n m) -> k n m', n=embed_q.size(0))

        scores = F.sigmoid(score + self.bias)

        return scores
