import os

import torch
import tqdm
from torch_geometric.datasets import TUDataset

from inari.utils import (
    feature_trans_categorical,
    k_subgraph,
    cal_matching_matrix,
    fix_random_seed,
)

fix_random_seed(42)

name = 'COX2'

dataset = TUDataset(root="raw", name=f"{name}")

targets = []
queries = []
mms = []

for t in tqdm.tqdm(dataset):
    if t.has_isolated_nodes():
        continue

    t = feature_trans_categorical(t)
    q = k_subgraph(t, int(0.3 * t.num_nodes))

    mm = cal_matching_matrix(t, q)

    if mm is None:
        continue

    targets.append(t)
    queries.append(q)
    mms.append(mm)

os.makedirs("data", exist_ok=True)
torch.save([targets, queries, mms], f"data/{name.lower()}.pt")