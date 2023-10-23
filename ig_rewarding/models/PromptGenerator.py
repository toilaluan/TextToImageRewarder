from transformers import pipeline, AutoModel, AutoTokenizer
import torch.nn as nn
from tqdm import tqdm
import torch
from urllib.request import urlretrieve
import pandas as pd
import chromadb
from typing import List
from chromadb.utils import embedding_functions

class PromptGenerator(nn.Module):
    def __init__(self, chroma_db_config):
        super().__init__()
        self.n_neighbors = chroma_db_config['n_neighbors']
        self.init_prompt_database(chroma_db_config)
        self.prompt_generation_pipe = pipeline("text-generation", model="Gustavosta/MagicPrompt-Stable-Diffusion")

    def download_prompt_data(self, url: str):
        urlretrieve(url, 'metadata.parquet')
        prompts = pd.read_parquet('metadata.parquet')['prompt'].tolist()
        return prompts
    def init_prompt_database(self, chroma_config):
        chroma_client = chromadb.PersistentClient(
            path=chroma_config['root_path'],
        )

        self.collection = chroma_client.get_or_create_collection(
            name=chroma_config['collection_name'], 
            metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 20, "hnsw:M": 5},
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                device="cuda", normalize_embeddings=False
            )
        )
        if self.collection.count() > 0:
            print("Prompt database already exists. Skipping creation.")
            return
        print("Downloading prompt data from", chroma_config['prompt_table_url'])
        prompts = self.download_prompt_data(chroma_config['prompt_table_url'])
        print("Adding prompts to database")
        ids = list(range(len(prompts)))
        ids = [str(id) for id in ids]
        batch_size = 512
        for i in tqdm(range(0, len(prompts), batch_size), total=len(prompts)//batch_size):
            batch = prompts[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            self.collection.add(documents=batch, ids=batch_ids)
        print("Building prompt database")

    def _retrieve_prompts(self, queries: List[str]) -> List[str]:
        k_prompts = self.collection.query(query_texts=queries, n_results=self.n_neighbors)
        distances = torch.tensor(k_prompts['distances'])
        similarity_scores = 1-distances
        index = torch.multinomial(similarity_scores, num_samples=1) # Shape N_query, 1
        index = index.squeeze(dim=1).tolist()
        prompts = []
        for query_index, prompt_index in enumerate(index):
            prompt = k_prompts['documents'][query_index]
            prompts.append(prompt[prompt_index])
        return prompts
    
    def generate_prompt(self, topics: List[str]):
        prompts = self._retrieve_prompts(topics)
        prompts = [f"{topic}, {prompt}" for topic, prompt in zip(topics, prompts)]
        generated_prompts = self.prompt_generation_pipe(prompts, min_length=32, max_length=77)
        generated_prompts = [prompt[0]['generated_text'] for prompt in generated_prompts]
        generated_prompts = [prompt.replace(f"{prefix_prompt}, ", "") for prompt, prefix_prompt in zip(generated_prompts, topics)]
        return generated_prompts


