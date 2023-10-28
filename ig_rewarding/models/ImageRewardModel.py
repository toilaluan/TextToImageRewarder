import torch.nn as nn
import ImageReward as IR
import torch
from PIL import Image
from typing import List


class ImageRewardModel(nn.Module):
    def __init__(self, name, device):
        super().__init__()
        # scale the reward from [-2, 2] to [0, 1]
        self.min = -2
        self.max = 2
        self.rm = IR.load(name, device=device)

    def forward(
        self, images: List[Image.Image], prompt: str, *args, **kwargs
    ) -> torch.FloatTensor:
        _, scores = self.rm.inference_rank(prompt, images)
        scores = torch.tensor(scores).mean()
        return (scores - self.min) / (self.max - self.min)
