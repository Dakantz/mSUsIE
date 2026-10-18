import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from .deps import engine
from .model import Clipart
from .routes import clipart, search

app = FastAPI()

with Session(engine) as session:
    cliparts = session.scalars(select(Clipart)).all()
    print(f"Loaded {len(cliparts)} cliparts from the database.")

clipart_dir = os.getenv("CLIPART_DIR", "/data/cliparts")
static_dir = Path(__file__).parent / "static"

app.mount("/cliparts", StaticFiles(directory=clipart_dir), name="cliparts")

app.include_router(search.router)
app.include_router(clipart.router)

# must stay last: catch-all static mount shadows anything registered after it
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
