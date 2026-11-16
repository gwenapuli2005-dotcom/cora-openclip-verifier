"""CORA image-relevance checker, backed by OpenCLIP.

This service does not replace MobileNetV3-Small and is not a trained out-of-scope
classifier. Calibrate thresholds on labeled local CORA images before enabling
automatic rejection.
"""

import io
import os
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

import open_clip
import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

MODEL_NAME = os.getenv("OPENCLIP_MODEL", "ViT-B-32")
PRETRAINED = os.getenv("OPENCLIP_WEIGHTS", "laion2b_s34b_b79k")
API_TOKEN = os.getenv("API_TOKEN", "")
AUTO_REJECT_ENABLED = os.getenv("AUTO_REJECT_ENABLED", "false").lower() == "true"
MIN_RELEVANCE_SCORE = os.getenv("MIN_RELEVANCE_SCORE")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "6"))
MAX_IMAGE_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# This is an image-text *similarity* checker, not an image caption generator.
CATEGORY_PROMPTS = {
    "Improper Waste Disposal": [
        "A photograph of household garbage illegally dumped on a roadside.",
        "A photo showing piles of scattered trash and litter in a public area.",
        "A picture of improperly discarded waste bags and rubbish outdoors.",
    ],
    "Streetlight Problem": [
        "A photograph of a damaged streetlight mounted on a pole.",
        "A public road streetlight with a visibly broken light fixture.",
        "A street lamp that appears damaged and in need of repair.",
    ],
    "Drainage/Flooding Concern": [
        "A photograph of a clogged roadside drainage canal.",
        "A flooded street with standing water near drainage channels.",
        "A photo showing a blocked gutter with water accumulation.",
    ],
    "Road Damage": [
        "A photograph of a pothole on a road surface.",
        "A concrete road with large visible cracks and damaged pavement.",
        "An asphalt road with broken pavement and damaged surface.",
    ],
}

# Representative negatives help with screening, but do NOT guarantee robust
# open-set detection against arbitrary unrelated images.
UNRELATED_PROMPTS = [
    "A selfie portrait photograph of a person.",
    "A close-up photograph of a plate of food.",
    "A picture of a pet cat or dog indoors.",
    "A photograph inside a clean bedroom or living room.",
    "A screenshot of a phone app or a written document.",
    "A photograph of a beautiful mountain or beach landscape.",
]

state = {}


def _embed_texts(model, tokenizer, prompts):
    tokens = tokenizer(prompts)
    with torch.inference_mode():
        features = model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_TOKEN:
        raise RuntimeError("Set API_TOKEN in the Render environment first.")
    if AUTO_REJECT_ENABLED and MIN_RELEVANCE_SCORE is None:
        raise RuntimeError(
            "Set a validated MIN_RELEVANCE_SCORE before AUTO_REJECT_ENABLED=true."
        )
    if MIN_RELEVANCE_SCORE is not None:
        threshold = float(MIN_RELEVANCE_SCORE)
        if not -1 <= threshold <= 1:
            raise RuntimeError("MIN_RELEVANCE_SCORE must be between -1 and 1")

    torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "2"))))
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device="cpu"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    category_vectors = []
    for prompts in CATEGORY_PROMPTS.values():
        prompt_vectors = _embed_texts(model, tokenizer, prompts)
        averaged = prompt_vectors.mean(dim=0)
        category_vectors.append(averaged / averaged.norm())

    state.update({
        "model": model,
        "preprocess": preprocess,
        "categories": list(CATEGORY_PROMPTS.keys()),
        "category_vectors": torch.stack(category_vectors),
        "unrelated_vectors": _embed_texts(model, tokenizer, UNRELATED_PROMPTS),
    })
    yield
    state.clear()


app = FastAPI(title="CORA OpenCLIP verifier", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "ready": bool(state)}


@app.post("/verify")
async def verify(
    file: Annotated[UploadFile, File(description="Camera image or testing image")],
    predicted_category: Annotated[str, Form(description="Actual MobileNet predicted label")],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
):
    if not x_api_key or not secrets.compare_digest(x_api_key, API_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if predicted_category not in CATEGORY_PROMPTS:
        raise HTTPException(status_code=422, detail="Unknown CORA category")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Only JPG, PNG, or WEBP images supported")

    raw = await file.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image file is too large")

    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > 15_000_000:
                raise HTTPException(status_code=413, detail="Image resolution too large")
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image file") from None

    tensor = state["preprocess"](image).unsqueeze(0)
    with torch.inference_mode():
        image_features = state["model"].encode_image(tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        category_scores = (image_features @ state["category_vectors"].T)[0]
        unrelated_scores = (image_features @ state["unrelated_vectors"].T)[0]

    names = state["categories"]
    score_map = {name: round(float(category_scores[i]), 4) for i, name in enumerate(names)}
    best_index = int(torch.argmax(category_scores))
    best_category = names[best_index]
    supplied_score = float(category_scores[names.index(predicted_category)])
    unrelated_score = float(torch.max(unrelated_scores))
    category_matches = (best_category == predicted_category)
    unrelated_competes = (unrelated_score >= supplied_score)
    needs_review = not category_matches or unrelated_competes

    # Advisory mode is the safe default. Do not equate similarity with calibrated
    # probability or silently use an arbitrary auto-rejection threshold.
    if not AUTO_REJECT_ENABLED:
        decision = "review_only"
    else:
        threshold = float(MIN_RELEVANCE_SCORE)
        decision = (
            "rejected"
            if needs_review or supplied_score < threshold
            else "accepted"
        )

    return {
        "predicted_category": predicted_category,
        "openclip_best_category": best_category,
        "category_matches": category_matches,
        "unrelated_example_competes": unrelated_competes,
        "predicted_category_similarity": round(supplied_score, 4),
        "highest_unrelated_similarity": round(unrelated_score, 4),
        "category_similarities": score_map,
        "decision": decision,
        "auto_reject_enabled": AUTO_REJECT_ENABLED,
        "note": "Similarity scores are NOT confidence probabilities. Validate thresholds on local data.",
    }
