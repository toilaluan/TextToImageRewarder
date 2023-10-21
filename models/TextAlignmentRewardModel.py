import torch.nn as nn
import ImageReward as IR
import torch


class TextAlignmentRewardModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.rm = IR.load(name)

    def forward(self, images, prompt):
        _, scores = self.rm.inference_rank(prompt, images)
        scores = torch.tensor(scores)
        return scores
