import torch
import torch.nn.functional as F
class SoftLabelCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        logprobs = F.log_softmax(x, dim=-1)
        loss = -(logprobs * y).sum(dim=-1)
        return loss.mean()