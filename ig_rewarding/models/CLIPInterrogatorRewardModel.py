import torch.nn as nn
from clip_interrogator import Config, Interrogator
from PIL import Image
import torch
from typing import List
from bert_score import BERTScorer


class CLIPInterrogatorRewardModel(nn.Module):
    def __init__(self, ci_config, bert_scorer_config, device="cuda"):
        super().__init__()
        ci_config["device"] = device
        bert_scorer_config["device"] = device
        self.device = ci_config["device"]
        self.interrogator = Interrogator(Config(**ci_config))
        self.bert_scorer = BERTScorer(**bert_scorer_config)

    def forward(
        self,
        images: List[Image.Image],
        prompt: str,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        ci_prompts = [self.interrogator.interrogate(image) for image in images]
        P_mul, R_mul, F_mul = self.bert_scorer.score(
            [prompt] * len(ci_prompts),
            ci_prompts,
            verbose=False,
        )
        return F_mul.mean()
