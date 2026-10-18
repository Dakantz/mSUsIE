from dataclasses import dataclass

import numpy as np
import umap
from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..deps import engine
from ..model import Clipart, ClipartData

router = APIRouter(prefix="/clipart")


def _data(clipart: Clipart) -> ClipartData:
    return ClipartData(
        path=f"/cliparts/{clipart.path}",
        name=clipart.name,
        clipart_id=clipart.clipart_id,
    )


@router.get("/{clipart_id}")
def get_clipart(clipart_id: int) -> ClipartData:
    with Session(engine) as session:
        clipart = session.get(Clipart, clipart_id)
        if clipart is None:
            raise HTTPException(status_code=404, detail="Clipart not found")
        return _data(clipart)


@router.get("/{clipart_id}/similar")
def similar_cliparts(clipart_id: int, limit: int = 20) -> list[ClipartData]:
    with Session(engine) as session:
        clipart = session.get(Clipart, clipart_id)
        if clipart is None:
            raise HTTPException(status_code=404, detail="Clipart not found")
        cliparts = session.scalars(
            select(Clipart)
            .where(Clipart.clipart_id != clipart_id)
            .order_by(Clipart.img_embedding.cosine_distance(clipart.img_embedding))
            .limit(limit)
        )
        return [_data(c) for c in cliparts]


@dataclass
class NeighbourhoodPoint:
    clipart_id: int
    name: str
    path: str
    x: float
    y: float


@router.get("/{clipart_id}/neighbourhood")
def neighbourhood(clipart_id: int, limit: int = 500) -> list[NeighbourhoodPoint]:
    """2D UMAP projection of the `limit` nearest cliparts, the selected one included."""
    with Session(engine) as session:
        clipart = session.get(Clipart, clipart_id)
        if clipart is None:
            raise HTTPException(status_code=404, detail="Clipart not found")
        cliparts = session.scalars(
            select(Clipart)
            .order_by(Clipart.img_embedding.cosine_distance(clipart.img_embedding))
            .limit(limit)
        ).all()

    # ponytail: UMAP on 500x512 costs a few seconds per request, recomputed every time.
    # Cache per clipart_id (or precompute a global projection at ingest) if it ever matters.
    xy = umap.UMAP(n_components=2, metric="cosine").fit_transform(
        np.array([c.img_embedding for c in cliparts])
    )
    return [
        NeighbourhoodPoint(c.clipart_id, c.name, f"/cliparts/{c.path}", float(x), float(y))
        for c, (x, y) in zip(cliparts, xy)
    ]
