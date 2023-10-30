import torch.nn as nn
import torch
from hpsv2.img_score import *
from PIL import Image
import torch
from typing import List
import numpy as np


class HPSv2(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model, self.preprocess, self.tokenizer = self._init_model()
    @torch.inference_mode()
    def forward(self, images: List[Image.Image], prompt: str, *args, **kwargs):
        # Process the image
        rewards = []
        images = [self.preprocess(image) for image in images]
        images = torch.stack(images).to(device=self.device, non_blocking=True)
        prompt = self.tokenizer([prompt])
        prompt = prompt.to(device=self.device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs = self.model(images, prompt)
            num_images = torch.tensor(images.size(0))
            image_features, text_features, logit_scale = outputs["image_features"], outputs["text_features"], outputs["logit_scale"]
            logits_per_image = image_features @ text_features.T
        return logits_per_image.mean()

    def _init_model(self, cp: str = os.path.join(root_path, "HPS_v2_compressed.pt")):
        initialize_model()
        model = model_dict["model"]
        preprocess_val = model_dict["preprocess_val"]

        # check if the checkpoint exists
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        if cp == os.path.join(root_path, "HPS_v2_compressed.pt") and not os.path.exists(
            cp
        ):
            print("Downloading HPS_v2_compressed.pt ...")
            url = "https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt"
            r = requests.get(url, stream=True)
            with open(os.path.join(root_path, "HPS_v2_compressed.pt"), "wb") as HPSv2:
                total_length = int(r.headers.get("content-length"))
                for chunk in progress.bar(
                    r.iter_content(chunk_size=1024),
                    expected_size=(total_length / 1024) + 1,
                ):
                    if chunk:
                        HPSv2.write(chunk)
                        HPSv2.flush()
            print(
                "Download HPS_2_compressed.pt to {} sucessfully.".format(
                    root_path + "/"
                )
            )

        checkpoint = torch.load(cp, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        tokenizer = get_tokenizer("ViT-H-14")
        model = model.to(self.device)
        model.eval()
        return model, preprocess_val, tokenizer
