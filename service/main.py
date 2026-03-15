from flask import Flask
from flask import request, Response, jsonify
from PIL import Image
import base64
from sentence_transformers import SentenceTransformer
from io import BytesIO
import re

device = "mps"
img_model = SentenceTransformer("clip-ViT-B-32").to(device)
text_model = SentenceTransformer(
    "sentence-transformers/clip-ViT-B-32-multilingual-v1"
).to(device)

milvus_connect()
# connect_collections()

cliparts = Collection("cliparts", enable_dynamic_field=True)

cliparts.load()

app = Flask(
    __name__,
    static_url_path="",
    static_folder="static",
)


@app.route("/query-image", methods=["POST"])
def query_img():
    if request.method == "POST":
        data: dict[str, any] = request.get_json()
        query = data["query"]
        weight = float(data["img_weight"])
        limit = data.get("limit", 20)

        query_embedding = text_model.encode([query]).squeeze()
        if "image" in data.keys():
            img_base_64 = data["image"]
            image_data = base64.b64decode(
                re.sub("^data:image/.+;base64,", "", img_base_64)
            )
            img = Image.open(BytesIO(image_data))

            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
            img = background
            # img.save("query.png")
            img_embedding = img_model.encode([img]).squeeze()
            embedding = query_embedding * (1.0 - weight) + img_embedding * weight
        else:
            embedding = query_embedding

        result = cliparts.search(
            [embedding],
            anns_field="img_embeddings",
            limit=limit,
            param={"metric_type": "COSINE", "params": {"nprobe": 2}},
            output_fields=["path", "text"],
        )

        result = [hit.fields for hit in result[0]]
        return jsonify(result)
