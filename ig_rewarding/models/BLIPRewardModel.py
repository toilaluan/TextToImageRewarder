import torch
import torch.nn as nn
from PIL import Image
from typing import List
import ImageReward as IR


class BLIPRewardModel(nn.Module):
    def __init__(self, name, device):
        super().__init__()
        self.rm = IR.load_score(name=name, device=device)

    def forward(
        self, images: List[Image.Image], prompt: str, *args, **kwargs
    ) -> torch.FloatTensor:
        _, scores = self.rm.inference_rank(prompt, images)
        scores = torch.tensor(scores).mean()
        return scores
