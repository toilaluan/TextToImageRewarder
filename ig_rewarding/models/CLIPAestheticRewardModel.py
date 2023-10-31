import torch
import torch.nn as nn
from typing import List
from PIL import Image
from pathlib import Path
import clip
from ig_rewarding.utils import download_file_to_cache


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class CLIPAestheticRewardModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.predictor, self.clip_model, self.clip_preprocess = self._prepare_model(
            self.device
        )

    def _prepare_model(self, device):
        state_name = "sac+logos+ava1-l14-linearMSE.pth"
        cache_folder = ".cache"
        ckpt_path = f"{cache_folder}/{state_name}"
        url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{state_name}?raw=true"
        download_file_to_cache(url, cache_folder, state_name)
        pt_state = torch.load(ckpt_path, map_location=torch.device("cpu"))
        predictor = AestheticPredictor(768)
        predictor.load_state_dict(pt_state)
        predictor.to(device)
        predictor.eval()
        clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
        return predictor, clip_model, clip_preprocess

    def get_image_features(self, images: List[Image.Image]):
        images = [self.clip_preprocess(image) for image in images]
        images = torch.stack(images).to(device=self.device, non_blocking=True)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    @torch.inference_mode()
    def forward(self, images: List[Image.Image], *args, **kwargs):
        image_features = self.get_image_features(images)
        score = self.predictor(image_features.float())
        return score.mean()


if __name__ == "__main__":
    model = CLIPAestheticRewardModel("cuda")
    images = [Image.new("RGB", (224, 224), color=(255, 0, 0))] * 4
    score = model(images)
