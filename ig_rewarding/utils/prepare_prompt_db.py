import chromadb
import pandas as pd
from urllib.request import urlretrieve

# Download the parquet table
PROMPT_TABLE_URL = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet'
urlretrieve(PROMPT_TABLE_URL, 'metadata.parquet')
prompts = pd.read_parquet('metadata.parquet')['prompt'].tolist()


chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="prompt_db")



