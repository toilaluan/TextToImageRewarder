from transformers import pipeline, AutoModel, AutoTokenizer
import torch.nn as nn
from tqdm import tqdm
import torch
from urllib.request import urlretrieve
import pandas as pd
import random
import chromadb
from typing import List
from chromadb.utils import embedding_functions


class PromptGenerator(nn.Module):
    def __init__(self, chroma_db_config):
        super().__init__()
        self.n_neighbors = chroma_db_config["n_neighbors"]
        self.init_prompt_database(chroma_db_config)

    def download_prompt_data(self, url: str):
        urlretrieve(url, "metadata.parquet")
        prompts = pd.read_parquet("metadata.parquet")["prompt"].tolist()
        prompts = list(set(prompts))
        return prompts

    def init_prompt_database(self, chroma_config):
        chroma_client = chromadb.PersistentClient(
            path=chroma_config["root_path"],
        )

        self.collection = chroma_client.get_or_create_collection(
            name=chroma_config["collection_name"],
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                device="cuda",
                normalize_embeddings=False,
                model_name="all-mpnet-base-v2",
            ),
        )
        if self.collection.count() > 0:
            print("Prompt database already exists. Skipping creation.")
            return
        print("Downloading prompt data from", chroma_config["prompt_table_url"])
        prompts = self.download_prompt_data(chroma_config["prompt_table_url"])
        print("Adding prompts to database")
        ids = list(range(len(prompts)))
        ids = [str(id) for id in ids]
        batch_size = 512
        for i in tqdm(
            range(0, len(prompts), batch_size), total=len(prompts) // batch_size
        ):
            batch = prompts[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            self.collection.add(documents=batch, ids=batch_ids)
        print("Building prompt database")

    def _retrieve_prompts(self, queries: List[str], n_prompts: int) -> List[str]:
        k_prompts = self.collection.query(query_texts=queries, n_results=n_prompts)
        return k_prompts["documents"]

    @torch.inference_mode()
    def generate_prompt(self, prefix_prompts: List[str], n_prompts: int = 100):
        generated_prompts = self._retrieve_prompts(prefix_prompts, n_prompts)
        return generated_prompts
