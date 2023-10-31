import torch.nn as nn
from PIL import Image
import torch
from typing import List
from ram.models import ram_plus
from ram import get_transform
from ig_rewarding.utils import download_file_to_cache


ckpt_file_link = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth"


class ImageTagMatchingRewardModel(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.model, self.transform = self._prepare_model(device)

    def _prepare_model(self, device):
        transform = get_transform(image_size=384)
        cache_folder = ".cache"
        model_name = "ram_plus_swin_large_14m.pth"
        ckpt_path = f"{cache_folder}/{model_name}"
        download_file_to_cache(ckpt_file_link, cache_folder, model_name)
        model = ram_plus(pretrained=ckpt_path, image_size=384, vit="swin_l")
        model.to(device)
        model.eval()
        return model, transform

    def forward(
        self,
        images: List[Image.Image],
        prompt: str,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        images = [self.transform(image) for image in images]
        images = torch.stack(images).to(device=self.device, non_blocking=True)
        tags, _ = self.model.generate_tag(images)
        print(tags)


if __name__ == "__main__":
    device = "cuda"
    model = ImageTagMatchingRewardModel(device)
    images = [Image.new("RGB", (384, 384)) for _ in range(2)]
    model(images, "test")
