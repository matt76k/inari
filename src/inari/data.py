import torch
from einops import reduce
from torch import BoolTensor, Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class MyDataset(Dataset):
    def __init__(self, root, num_features):
        super().__init__()
        self.target, self.query, self.mm = torch.load(root, weights_only=False)

        self.num_features = num_features

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        t, q, mm = self.target[idx], self.query[idx], self.mm[idx]

        mm = reduce(mm, "c q t -> q t", "max")

        return t, q, mm


def batch_mm(labels: list[Tensor]) -> tuple[Tensor, BoolTensor]:
    batch = []
    n, m = 0, 0
    for mm in labels:
        n += mm.size(0)
        m += mm.size(1)
        batch.append((mm, n, m))

    bmm = torch.zeros(batch[-1][1], batch[-1][2], dtype=torch.float32)
    mask = torch.zeros(batch[-1][1], batch[-1][2], dtype=torch.bool)

    for b, i, j in batch:
        i_0 = i - b.size(0)
        j_0 = j - b.size(1)
        bmm[i_0:i, j_0:j] = b
        mask[i_0:i, j_0:j] = True

    return bmm, mask


def prepare_data_for_subgraph_task(samples) -> tuple[Batch, Batch, Tensor, BoolTensor]:
    t, q, labels = map(list, zip(*samples))
    target = Batch.from_data_list(t)
    query = Batch.from_data_list(q)

    mm, mask = batch_mm(labels)
    return target, query, mm, mask
