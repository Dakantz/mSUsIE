import base64
import re
from io import BytesIO

from fastapi import APIRouter, Body
from fastapi.responses import Response
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..deps import engine, img_model, text_model
from ..model import Clipart, ClipartData

router = APIRouter()


@router.post("/query-image")
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
        image_data = base64.b64decode(re.sub("^data:image/.+;base64,", "", image))
        img = Image.open(BytesIO(image_data))

        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        img = background
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
        return [
            ClipartData(
                path=f"/cliparts/{clipart.path}",
                name=clipart.name,
                clipart_id=clipart.clipart_id,
            )
            for clipart in cliparts
        ]
