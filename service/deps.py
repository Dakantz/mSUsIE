"""Shared singletons: DB engine and the two CLIP models. Imported by main.py and the routers."""

import os

import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine

conn_str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/postgres")
engine = create_engine(conn_str)

dev = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
img_model = SentenceTransformer("clip-ViT-B-32").to(dev)
text_model = SentenceTransformer(
    "sentence-transformers/clip-ViT-B-32-multilingual-v1"
).to(dev)
