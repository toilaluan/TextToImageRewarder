# Fooocus GPT2 Expansion
# Algorithm created by Lvmin Zhang at 2023, Stanford
# If used inside Fooocus, any use is permitted.
# If used outside Fooocus, only non-commercial use is permitted (CC-By NC 4.0).
# This applies to the word list, vocab, model, and algorithm.


import os
import torch.nn as nn
import torch
import math
from transformers.generation.logits_process import LogitsProcessorList
from transformers import AutoTokenizer, AutoModelForCausalLM
from urllib.parse import urlparse
from typing import Optional

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


# limitation of np.random.seed(), called from transformers.set_seed()
SEED_LIMIT_NUMPY = 2**32
neg_inf = -8192.0


def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file

        download_url_to_file(url, cached_file, progress=progress)
    return cached_file


def safe_str(x):
    x = str(x)
    for _ in range(16):
        x = x.replace("  ", " ")
    return x.strip(",. \r\n")


def remove_pattern(x, pattern):
    for p in pattern:
        x = x.replace(p, "")
    return x


class FooocusExpansion(nn.Module):
    def __init__(self, url, device):
        super().__init__()
        fooocus_expansion_path = f"{DIR_PATH}/fooocus_expansion"
        load_file_from_url(
            url=url,
            model_dir=fooocus_expansion_path,
            file_name="pytorch_model.bin",
        )
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(fooocus_expansion_path)

        positive_words = (
            open(
                os.path.join(DIR_PATH, "fooocus_expansion/positive.txt"),
                encoding="utf-8",
            )
            .read()
            .splitlines()
        )
        positive_words = ["Ġ" + x.lower() for x in positive_words if x != ""]

        self.logits_bias = (
            torch.zeros((1, len(self.tokenizer.vocab)), dtype=torch.float32) + neg_inf
        )

        debug_list = []
        for k, v in self.tokenizer.vocab.items():
            if k in positive_words:
                self.logits_bias[0, v] = 0
                debug_list.append(k[1:])

        print(f"Fooocus V2 Expansion: Vocab with {len(debug_list)} words.")

        self.model = AutoModelForCausalLM.from_pretrained(fooocus_expansion_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    @torch.inference_mode()
    def logits_processor(self, input_ids, scores):
        assert scores.ndim == 2 and scores.shape[0] == 1
        self.logits_bias = self.logits_bias.to(scores)

        bias = self.logits_bias.clone()
        bias[0, input_ids[0].to(bias.device).long()] = neg_inf
        bias[0, 11] = 0

        return scores + bias

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, prompt):
        if prompt == "":
            return ""
        prompt = safe_str(prompt) + ","

        tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
        tokenized_kwargs.data["input_ids"] = tokenized_kwargs.data["input_ids"].to(
            self.device
        )
        tokenized_kwargs.data["attention_mask"] = tokenized_kwargs.data[
            "attention_mask"
        ].to(self.device)

        current_token_length = int(tokenized_kwargs.data["input_ids"].shape[1])
        max_token_length = 75 * int(math.ceil(float(current_token_length) / 75.0))
        max_new_tokens = max_token_length - current_token_length

        # https://huggingface.co/blog/introducing-csearch
        # https://huggingface.co/docs/transformers/generation_strategies
        features = self.model.generate(
            **tokenized_kwargs,
            top_k=100,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            logits_processor=LogitsProcessorList([self.logits_processor]),
        )

        response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
        result = safe_str(response[0])

        return result


if __name__ == "__main__":
    url = "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin"
    device = "cuda"
    fooocus_expansion = FooocusExpansion(url, device)
    prompt = "a animated image"
    seed = 0
    result = fooocus_expansion(prompt)
    print(result)
