import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import BoolTensor, Tensor


class MMLoss(nn.Module):
    def __init__(self, criterion=nn.MSELoss(), mode="all"):
        super().__init__()
        self.criterion = criterion
        self.mode = mode

    def forward(self, matching_matrix: Tensor, label: Tensor, mask: BoolTensor) -> Tensor:
        if self.mode == "last":
            mm = matching_matrix * mask.to(torch.float32)
            return self.criterion(mm[0], label)

        if self.mode == "mse":
            mm = matching_matrix * mask.to(torch.float32)
            mm = rearrange(mm, "c q t -> (c q) t")
            return self.criterion(mm, repeat(label, "q t -> (c q) t", c=matching_matrix.size(0)))

        if self.mode == "default":
            mm_0 = matching_matrix * (label == 0).to(torch.float32) * mask.to(torch.float32)
            mm_0 = rearrange(mm_0, "c q t -> (c q) t")
            mm_0 = reduce(mm_0, "n t -> n 1", reduction="sum")

            matching_label = torch.zeros(mm_0.size(0), 1, dtype=torch.float32)

            return self.criterion(mm_0, matching_label)

        mm_0 = matching_matrix * (label == 0).to(torch.float32) * mask.to(torch.float32)
        mm_0 = rearrange(mm_0, "c q t -> (c q) t")
        mm_0 = reduce(mm_0, "n t -> n 1", reduction="sum")
        mm_1 = matching_matrix * (label == 1).to(torch.float32) * mask.to(torch.float32)
        mm_1 = rearrange(mm_1, "c q t -> (c q) t")
        mm_1 = reduce(mm_1, "n t -> n 1", reduction="sum")

        matching_label = torch.ones(mm_0.size(0), 1, dtype=torch.float32)

        return self.criterion(mm_1 - mm_0, matching_label)


class PermutationLoss(nn.Module):
    def __init__(self):
        super(PermutationLoss, self).__init__()

    def forward(self, matching_matrix: Tensor, label: Tensor, mask: BoolTensor):
        mm = matching_matrix * mask.to(torch.float32)
        mm = rearrange(mm, "c q t -> (c q) t")

        return F.binary_cross_entropy(mm, repeat(label, "q t -> (c q) t", c=matching_matrix.size(0)))
