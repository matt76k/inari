from typing import Tuple

import torch
import torch_geometric.transforms as T
from einops import reduce
from torch import Tensor
from torch_geometric.data import Data

from .utils import gen_subgraph, mk_matching_matrix


class GenSubgraphMM(T.BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        t: Data,
    ) -> Tuple[Data, Data, Tensor]:
        q = gen_subgraph(t)
        mm = mk_matching_matrix(t, q)
        mm = reduce(mm, "c q t -> q t", "max")

        t.x = torch.argmax(t.x, dim=1)
        q.x = torch.argmax(q.x, dim=1)

        return t, q, mm

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
