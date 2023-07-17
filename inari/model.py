import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

from .layers import NTN, SimpleMM


def get_att(embed_t, embed_q, mask):
    fmask = mask.to(torch.float32)

    dim = embed_t.size(1)
    att = torch.mm(embed_q, embed_t.T)
    att *= fmask
    att = att / (dim**0.5)
    att += -1e9 * (~mask).to(torch.float32)
    att = F.softmax(att, dim=1)

    return att


class SimpleSubGMN(nn.Module):
    def __init__(self, num_features: int, h_dim: int, num_layers=3):
        super().__init__()
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.project = nn.Embedding(num_features + 1, h_dim)
        self.GCN = nn.ModuleList([SAGEConv(h_dim, h_dim) for _ in range(num_layers)])
        self.activation = F.elu
        self.W = torch.nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, target, query, mask):
        embed_t = self.project(target.x)
        embed_q = self.project(query.x)

        for i in range(self.num_layers):
            embed_t = self.activation(self.GCN[i](embed_t, target.edge_index))
            embed_q = self.activation(self.GCN[i](embed_q, query.edge_index))

        mm = get_att(embed_t, embed_q, mask)

        return torch.unsqueeze(mm, 0)


class SubGMN(nn.Module):
    def __init__(self, num_features, h_dim, k, num_layers=3, feature_type: str = "categorical"):
        super().__init__()
        self.h_dim = h_dim
        self.k = k
        self.num_layers = num_layers

        self.project = (
            nn.Embedding(num_features + 1, h_dim) if feature_type == "categorical" else nn.Linear(num_features, h_dim)
        )

        self.GCN = nn.ModuleList([SAGEConv(h_dim, h_dim) for _ in range(num_layers)])
        self.NTNs = nn.ModuleList([NTN(h_dim, k) for _ in range(num_layers)])
        self.activation = [F.elu for _ in range(num_layers - 1)] + [F.sigmoid]  # なんで最後だけsigmoid
        self.last = nn.Conv2d(k * num_layers, 1, 1)

    def forward(self, target, query, mask):
        op_list = []
        embed_t = self.project(target.x)
        embed_q = self.project(query.x)

        for i in range(self.num_layers):
            embed_t = self.activation[i](self.GCN[i](embed_t, target.edge_index))
            embed_q = self.activation[i](self.GCN[i](embed_q, query.edge_index))
            att = get_att(embed_t, embed_q, mask)
            ntn = self.NTNs[i](embed_t, embed_q)
            op = ntn * att
            op_list.append(op)

        mm = torch.softmax(self.last(torch.cat(op_list)).squeeze(0), dim=1)

        return torch.unsqueeze(mm, 0)


class SubCrossGMN(nn.Module):
    def __init__(
        self,
        num_features,
        h_dim,
        num_layers=3,
        feature_type: str = "categorical",
        similarity_layer: nn.Module = SimpleMM,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.project = (
            nn.Embedding(num_features + 1, h_dim) if feature_type == "categorical" else nn.Linear(num_features, h_dim)
        )

        self.GCN = nn.ModuleList([SAGEConv(h_dim, h_dim) for _ in range(num_layers)])
        self.SIM = nn.ModuleList([similarity_layer(h_dim) for _ in range(num_layers)])

        self.coef_t = nn.Parameter(torch.empty((num_layers, 3)))
        self.coef_q = nn.Parameter(torch.empty((num_layers, 3)))
        nn.init.constant_(self.coef_t, 1 / 3)
        nn.init.constant_(self.coef_q, 1 / 3)

    def forward(self, target, query, mask):
        att_list = []
        embed_t = embed_t_0 = self.project(target.x)
        embed_q = embed_q_0 = self.project(query.x)

        for i in range(self.num_layers):
            embed_t = F.elu(self.GCN[i](embed_t, target.edge_index))
            embed_q = F.elu(self.GCN[i](embed_q, query.edge_index))

            att = self.SIM[i](embed_t, embed_q, mask)

            att_list.append(att)

            cross_t = torch.mm(att.T, embed_q)
            cross_q = torch.mm(att, embed_t)

            embed_q = self.coef_q[i, 0] * embed_q_0 + self.coef_q[i, 1] * embed_q + self.coef_q[i, 2] * cross_q
            embed_t = self.coef_t[i, 0] * embed_t_0 + self.coef_t[i, 1] * embed_t + self.coef_t[i, 2] * cross_t

        att = get_att(embed_t, embed_q, mask)
        att_list.append(att)

        return torch.stack(att_list)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)


class GGUnit(nn.Module):
    def __init__(self, h_dim: int):
        super().__init__()

        self.norm = nn.LayerNorm(h_dim)
        self.act = torch.tanh
        self.gcn = GCNConv(h_dim, h_dim)

        init_eps = 1e-3 / h_dim
        nn.init.uniform_(self.gcn.lin.weight, -init_eps, init_eps)
        nn.init.constant_(self.gcn.bias, 1.0)

    def forward(self, x, edge_index):
        gate = self.norm(x)
        gate = self.gcn(gate, edge_index)
        gate = self.act(gate)
        return gate * x


class gMLPLayer(nn.Module):
    def __init__(self, h_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(h_dim)
        self.proj_in = nn.Linear(h_dim, h_dim)
        self.ggu = GGUnit(h_dim)
        self.proj_out = nn.Linear(h_dim, h_dim)

    def forward(self, x, edge_index):
        x = self.norm(x)
        x = self.proj_in(x)
        x = F.gelu(x)
        x = self.ggu(x, edge_index)
        x = self.proj_out(x)
        return x


class g2MLP(nn.Module):
    def __init__(self, h_dim: int, num_layers=3):
        super().__init__()
        self.h_dim = h_dim
        self.layers = nn.ModuleList([Residual(gMLPLayer(h_dim)) for _ in range(num_layers)])

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)

        return x
