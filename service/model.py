from dataclasses import dataclass

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

Base = declarative_base()
N_EMBEDDINGS = 512


class Clipart(Base):
    __tablename__ = "cliparts"
    clipart_id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column()
    name: Mapped[str] = mapped_column()
    embedding: Mapped[Vector | None] = mapped_column(Vector(N_EMBEDDINGS))
    img_embedding: Mapped[Vector] = mapped_column(Vector(N_EMBEDDINGS))
    text_embedding: Mapped[Vector] = mapped_column(Vector(N_EMBEDDINGS))
    avg_embedding: Mapped[Vector] = mapped_column(Vector(N_EMBEDDINGS))


@dataclass
class ClipartData:
    path: str
    name: str
    clipart_id: int
