"""CORA OpenCLIP image-relevance verifier, ONNX Runtime edition.

MobileNetV3-Small remains CORA's official four-category classifier.
This endpoint returns advisory image-text similarity scores, NOT reliable
out-of-scope probabilities. Auto rejection defaults to OFF.
"""

import io
import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

BASE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE / "models/openclip_vision_int8.onnx")))
EMBEDDINGS_PATH = Path(os.getenv("EMBEDDINGS_PATH", str(BASE / "models/text_embeddings.json")))
API_TOKEN = os.getenv("API_TOKEN", "")
AUTO_REJECT_ENABLED = os.getenv("AUTO_REJECT_ENABLED", "false").lower() == "true"
# Calibrate these on labeled in-scope and out-of-scope images BEFORE auto rejection.
OOD_MIN_MARGIN = os.getenv("OOD_MIN_MARGIN")
OOD_MAX_CATEGORY_SCORE = os.getenv("OOD_MAX_CATEGORY_SCORE")
OOD_MIN_UNRELATED_SCORE = os.getenv("OOD_MIN_UNRELATED_SCORE")
MAX_IMAGE_BYTES = int(os.getenv("MAX_UPLOAD_MB", "6")) * 1024 * 1024
IMAGE_SIZE = 224
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

state = {}


def _unit(vectors):
    x = np.asarray(vectors, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _preprocess(image):
    # Match OpenCLIP ViT-B-32's RGB -> bicubic resize shortest side -> center crop.
    image = image.convert("RGB")
    width, height = image.size
    if width <= height:
        new_width = IMAGE_SIZE
        new_height = max(IMAGE_SIZE, int(IMAGE_SIZE * height / width))
    else:
        new_height = IMAGE_SIZE
        new_width = max(IMAGE_SIZE, int(IMAGE_SIZE * width / height))
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    left = (new_width - IMAGE_SIZE) // 2
    top = (new_height - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - CLIP_MEAN) / CLIP_STD
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None, ...], dtype=np.float32)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_TOKEN:
        raise RuntimeError("Set API_TOKEN in Render Environment first")
    if AUTO_REJECT_ENABLED:
        required = [OOD_MIN_MARGIN, OOD_MAX_CATEGORY_SCORE, OOD_MIN_UNRELATED_SCORE]
        if any(value is None for value in required):
            raise RuntimeError(
                "Do not enable auto reject without calibrated OOD_MIN_MARGIN, "
                "OOD_MAX_CATEGORY_SCORE, and OOD_MIN_UNRELATED_SCORE"
            )
    for name, value in (
        ("OOD_MIN_MARGIN", OOD_MIN_MARGIN),
        ("OOD_MAX_CATEGORY_SCORE", OOD_MAX_CATEGORY_SCORE),
        ("OOD_MIN_UNRELATED_SCORE", OOD_MIN_UNRELATED_SCORE),
    ):
        if value is not None and not -1.0 <= float(value) <= 1.0:
            raise RuntimeError(f"{name} must be between -1 and 1")

    if not MODEL_PATH.is_file() or not EMBEDDINGS_PATH.is_file():
        raise RuntimeError("Model files missing. Set MODEL_RELEASE_URL and run build.sh.")

    with EMBEDDINGS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    categories = payload["categories"]
    # The JSON is produced by export_onnx.py from the exact original prompts.
    names = list(categories)
    if names != [
        "Improper Waste Disposal", "Streetlight Problem",
        "Drainage/Flooding Concern", "Road Damage",
    ]:
        raise RuntimeError("Category order in text_embeddings.json is unexpected")
    category_vectors = _unit([categories[name] for name in names])
    unrelated_vectors = _unit(payload["unrelated"])
    if category_vectors.shape[1] != unrelated_vectors.shape[1]:
        raise RuntimeError("Text embedding dimensions differ")

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, int(os.getenv("ORT_THREADS", "1")))
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = False  # Lower peak memory, at a performance cost
    options.enable_mem_pattern = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    # Use one image per request to avoid memory spikes on 512 MB instances.
    output_shape = session.get_outputs()[0].shape
    if isinstance(output_shape[-1], int) and output_shape[-1] != category_vectors.shape[1]:
        raise RuntimeError("ONNX output and text embedding dimensions differ")
    state.update(session=session, input_name=input_name, categories=names,
                 category_vectors=category_vectors, unrelated_vectors=unrelated_vectors)
    yield
    state.clear()


app = FastAPI(
    title="CORA ONNX OpenCLIP Verifier", version="2.0.0", lifespan=lifespan,
    # Keep the endpoint schema off the public internet; /verify requires a key.
    docs_url=None, redoc_url=None, openapi_url=None,
)


@app.get("/health")
def health():
    return {"status": "ok", "ready": bool(state)}


@app.post("/verify")
async def verify(
    file: Annotated[UploadFile, File(description="Resident camera or authorized test image")],
    predicted_category: Annotated[str, Form(description="Actual MobileNetV3-Small prediction")],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
):
    if not x_api_key or not API_TOKEN or not secrets.compare_digest(x_api_key, API_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not state:
        raise HTTPException(status_code=503, detail="Model is not ready")
    if predicted_category not in state["categories"]:
        raise HTTPException(status_code=422, detail="Unknown CORA category")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Only JPG, PNG, and WEBP images supported")
    raw = await file.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the upload size limit")
    try:
        with Image.open(io.BytesIO(raw)) as src:
            if src.width * src.height > 15_000_000:
                raise HTTPException(status_code=413, detail="Image resolution too large")
            image = ImageOps.exif_transpose(src).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image file") from None

    tensor = _preprocess(image)
    features = state["session"].run(None, {state["input_name"]: tensor})[0]
    features = _unit(features)[0]
    scores = state["category_vectors"] @ features
    unrelated_scores = state["unrelated_vectors"] @ features

    names = state["categories"]
    best_category = names[int(np.argmax(scores))]
    supplied_score = float(scores[names.index(predicted_category)])
    unrelated_score = float(np.max(unrelated_scores))
    category_matches = best_category == predicted_category
    best_category_score = float(np.max(scores))
    unrelated_competes = unrelated_score >= supplied_score  # backward-compatible legacy field
    unrelated_exceeds_all_categories = unrelated_score > best_category_score
    unrelated_margin = unrelated_score - best_category_score

    # Important: disagreement between two CORA categories is NOT an out-of-scope finding.
    # For example, garbage in drainage may validly relate to both categories.
    if unrelated_exceeds_all_categories:
        verification_status = "out_of_scope_candidate"
    elif not category_matches:
        verification_status = "category_disagreement_needs_review"
    else:
        verification_status = "relevant_candidate"

    if not AUTO_REJECT_ENABLED:
        # All results remain advisory until calibrated on a labeled test set.
        decision = "review_only"
    else:
        # Only reject when THREE calibrated conditions indicate unrelated content.
        # Category disagreement alone is NEVER a reason to auto-reject.
        confirmed_ood = (
            unrelated_exceeds_all_categories
            and unrelated_margin >= float(OOD_MIN_MARGIN)
            and best_category_score <= float(OOD_MAX_CATEGORY_SCORE)
            and unrelated_score >= float(OOD_MIN_UNRELATED_SCORE)
        )
        if confirmed_ood:
            decision = "rejected"
        elif verification_status == "relevant_candidate":
            decision = "accepted"
        else:
            decision = "needs_review"

    return {
        "predicted_category": predicted_category,
        "openclip_best_category": best_category,
        "category_matches": category_matches,
        "unrelated_example_competes": unrelated_competes,
        "unrelated_exceeds_all_categories": unrelated_exceeds_all_categories,
        "unrelated_margin_vs_best_category": round(unrelated_margin, 4),
        "verification_status": verification_status,
        "needs_review": verification_status != "relevant_candidate",
        "predicted_category_similarity": round(supplied_score, 4),
        "highest_unrelated_similarity": round(unrelated_score, 4),
        "category_similarities": {name: round(float(value), 4) for name, value in zip(names, scores)},
        "decision": decision,
        "auto_reject_enabled": AUTO_REJECT_ENABLED,
        "note": "Advisory similarity scores, NOT probabilities. Confirm thresholds with labeled data before enabling auto reject.",
    }
