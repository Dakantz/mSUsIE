import base64
from pathlib import Path
import re
from io import BytesIO


import torch
from flask import Flask, Response, jsonify, request
from .model import Clipart, ClipartData
from PIL import Image
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from fastapi import Body, FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

import os

conn_str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/postgres")
engine = create_engine(conn_str)

with Session(engine) as session:
    cliparts = session.scalars(select(Clipart)).all()
    print(f"Loaded {len(cliparts)} cliparts from the database.")

clipart_dir = os.getenv("CLIPART_DIR", "/data/cliparts")
static_dir = Path(__file__).parent / "static"

app.mount("/cliparts", StaticFiles(directory=clipart_dir), name="cliparts")


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


@app.post("/query-image")
def query_img(
    image: str | None = Body(
        None, example="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    ),
    query: str | None = Body(None, example="A cat sitting on a mat"),
    weight: float = Body(0.5, example=0.5),
    limit: int = Body(20, example=20),
):
    if query is None and image is None:
        return Response(
            content="Either query or image must be provided", status_code=400
        )
    if query is None:
        query = ""
    query_embedding = text_model.encode([query]).squeeze()
    if image:
        img_base_64 = image
        image_data = base64.b64decode(re.sub("^data:image/.+;base64,", "", img_base_64))
        img = Image.open(BytesIO(image_data))

        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        img = background
        # img.save("query.png")
        img_embedding = img_model.encode([img]).squeeze()
        embedding = query_embedding * (1.0 - weight) + img_embedding * weight
    else:
        embedding = query_embedding
    with Session(engine) as session:
        cliparts = session.scalars(
            select(Clipart)
            .order_by(Clipart.img_embedding.cosine_distance(embedding))
            .limit(limit)
        )
        result = [
            ClipartData(
                path=f"/cliparts/{clipart.path}",
                name=clipart.name,
                clipart_id=clipart.clipart_id,
            )
            for clipart in cliparts
        ]

        return result


app.mount("/", StaticFiles(directory=static_dir), name="static")
