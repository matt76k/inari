import io
import random
import resource
import shutil
import subprocess
from tempfile import NamedTemporaryFile
from typing import IO

import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
from einops import rearrange
from networkx.algorithms.isomorphism import DiGraphMatcher, numerical_node_match
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, k_hop_subgraph, remove_isolated_nodes, to_networkx


def convert_vf(G: nx.Graph, named: IO, feature: str = "x"):
    vf = io.StringIO()
    vf.write(f"{G.number_of_nodes()}\n\n")

    for node in G.nodes:
        vf.write(f"{node} {G.nodes[node][feature]}\n")
    vf.write("\n")

    for node in G.nodes:
        vf.write(f"{G.out_degree(node)}\n")
        for src, tgt in G.out_edges(node):
            vf.write(f"{src} {tgt}\n")

    vf.seek(0)
    shutil.copyfileobj(vf, named)
    named.flush()


def setlimits():
    resource.setrlimit(resource.RLIMIT_AS, (500 * 1024**3, -1))


def mk_matching_matrix_vf3(target: Data, query: Data, feature: str = "x", timeout=10) -> Tensor:
    with NamedTemporaryFile(mode="w") as t, NamedTemporaryFile(mode="w") as q:
        convert_vf(to_networkx(target, node_attrs=[feature]), t, feature)
        convert_vf(to_networkx(query, node_attrs=[feature]), q, feature)

        try:
            proc = subprocess.Popen(
                ["./vf3lib/bin/vf3", "-s", f"{q.name}", f"{t.name}"], stdout=subprocess.PIPE, preexec_fn=setlimits
            )
            out, _ = proc.communicate(timeout=timeout)

            mm = out.decode().split("\n")[1:-2]
            matching_matrix = torch.zeros(len(mm), query.num_nodes, target.num_nodes, dtype=torch.float32)

            for c in range(len(mm)):
                for xy in mm[c].split(":")[:-1]:
                    y, x = map(int, xy.split(","))
                    matching_matrix[c, x, y] = 1.0
            return matching_matrix
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
            print(e)
            return None


def cal_matching_matrix(target: Data, query: Data, feature: str = "x") -> Tensor:
    """
    Matching Matrixを計算
    """
    t = to_networkx(target, node_attrs=[feature])
    q = to_networkx(query, node_attrs=[feature])
    matcher = DiGraphMatcher(t, q, node_match=numerical_node_match(feature, 1.0))
    mm = list(matcher.subgraph_isomorphisms_iter())

    points = np.array(list(map(lambda x: list(x.items()), mm)))

    chan = points.shape[0]

    matching_matrix = torch.zeros(chan, query.num_nodes, target.num_nodes, dtype=torch.float32)

    for c in range(chan):
        y = points[c][:, 0]
        x = points[c][:, 1]

        matching_matrix[c, x, y] = 1.0

    return matching_matrix


def gen_subgraph(graph: Data, min_ratio: float = 0.5, max_ratio: float = 0.7) -> Data:
    k = 1
    min_size = int(min_ratio * graph.num_nodes)
    max_size = int(max_ratio * graph.num_nodes)
    idx = random.randint(0, graph.num_nodes - 1)

    last = Data()
    last.num_nodes = 0
    cnt = 0

    while True:
        subset, edge_index, _, _ = k_hop_subgraph(idx, k, graph.edge_index, relabel_nodes=True)
        data = Data(x=graph.x[subset], edge_index=edge_index)

        if min_size < data.num_nodes < max_size:
            return data
        elif len(subset) > max_size:
            idx = random.randint(0, graph.num_nodes - 1)
            k = 1
            cnt += 1
        else:
            k += 1

        if last.num_nodes < data.num_nodes:
            last = data

        if k > 8 or cnt > graph.num_nodes:
            return last


def k_subgraph(graph: Data, num_nodes_s: int = 10, node_idx: int | None = None) -> Data:
    if node_idx is None:
        node_idx = random.randint(0, graph.num_nodes - 1)

    num_nodes = graph.num_nodes
    col, row = graph.edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    node_idx = torch.tensor([node_idx], device=row.device).flatten()

    subsets = [node_idx]
    subset = node_idx

    n = 1
    while True:
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])
        subset = torch.cat((subset, subsets[-1]), dim=0).unique()

        if n == subset.size(0):
            break

        n = subset.size(0)

        if n > num_nodes_s:
            subset = subset[:num_nodes_s]
            break

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = graph.edge_index[:, edge_mask]

    node_idx = row.new_full((num_nodes,), -1)
    node_idx[subset] = torch.arange(subset.size(0), device=row.device)
    edge_index = node_idx[edge_index]

    return Data(x=graph.x[subset], edge_index=edge_index)


def gen_not_subgraph(d: Data, ratio: float = 0.7) -> Data:
    num_nodes = int(d.num_nodes * ratio)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.2, directed=True)
    edge_index = remove_isolated_nodes(edge_index)[0]

    x = d.x[torch.randperm(d.num_nodes)[:num_nodes]]

    return Data(x=x, edge_index=edge_index)


def fix_random_seed(random_seed: int = 42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def metric_acc(mm: Tensor, label: Tensor) -> float:
    idx = torch.argmax(mm, dim=1)
    row = torch.arange(label.size(0))
    value = label[row, idx]
    acc = torch.sum(value) / len(value)
    return acc.detach().item()


def metric_f1(mm, label):
    y_pred = torch.argmax(mm, dim=1)
    y_pred = torch.nn.functional.one_hot(y_pred, num_classes=mm.size(1))
    y_pred = rearrange(y_pred, "n m -> (n m)")

    y_true = rearrange(label, "n m -> (n m)").to(y_pred.dtype)

    f1_score = metrics.f1_score(y_true, y_pred)
    return f1_score


def feature_trans_categorical(t: Data) -> Data:
    """
    one-hotになってる特徴量を数字へ変換
    """
    t.x = torch.argmax(t.x, dim=1)
    return t


def feature_trans_numerical(t: Data) -> tuple[Data, dict[str, int]]:
    features = {}
    xs = []
    for x in map(str, t.x.tolist()):
        n = features.get(x, None)
        if n is None:
            features[x] = len(features)
            xs.append(features[x])
        else:
            xs.append(n)

    t["numeric_x"] = torch.tensor(xs, dtype=torch.int32).unsqueeze(1)

    return t, features
