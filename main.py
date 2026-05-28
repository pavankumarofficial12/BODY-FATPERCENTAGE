import cv2
import numpy as np
import mediapipe as mp
import math
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from tenacity import retry, stop_after_attempt, wait_exponential
import redis.asyncio as redis
import uuid
from typing import List, Optional

# Initialize FastAPI and Redis
app = FastAPI(title="Production AI Body Fat Estimator (Body-Part Output)")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# Rate limit exception handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request, exc):
    return {"error": "Too many requests. Please try again later."}

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODELS
# =========================
mp_seg = mp.solutions.selfie_segmentation
segmenter = mp_seg.SelfieSegmentation(model_selection=1)

# =========================
# IMAGE VALIDATION
# =========================
def validate_image(image):
    h, w = image.shape[:2]
    if h < 400 or w < 250:
        raise HTTPException(400, "Image resolution too low")

# =========================
# SEGMENTATION
# =========================
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
async def segment_body(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = segmenter.process(rgb)
    if res.segmentation_mask is None:
        raise HTTPException(400, "Body not detected")
    return (res.segmentation_mask > 0.5).astype(np.uint8)

# =========================
# SILHOUETTE SLICING
# =========================
def extract_slices(mask, slices=16):
    rows = np.where(mask.sum(axis=1) > 25)[0]
    if len(rows) < 120:
        raise HTTPException(400, "Incomplete body visibility")

    top, bottom = rows[0], rows[-1]
    ys = np.linspace(top, bottom, slices)

    widths = []
    for y in ys:
        row = mask[int(y)]
        xs = np.where(row > 0)[0]
        widths.append(xs[-1] - xs[0] if len(xs) > 15 else 0)

    return bottom - top, widths

# =========================
# VOLUME ESTIMATION
# =========================
def estimate_volume(px_height, widths, height_cm, view):
    cm_per_px = height_cm / px_height
    slice_h = height_cm / len(widths)

    volume = 0.0
    for i, w_px in enumerate(widths):
        width_cm = w_px * cm_per_px
        width_cm *= 0.9  # clothing correction

        a = width_cm / 2
        b = a * (0.65 if view == "side" else 0.55)

        importance = 1.3 if 6 <= i <= 10 else 1.0
        volume += importance * math.pi * a * b * slice_h

    return volume

# =========================
# BODY FAT %
# =========================
def estimate_body_fat(volume, weight_kg, height_cm, gender):
    density = (weight_kg * 1000) / volume
    fat = (4.95 / density - (4.50 if gender == "male" else 4.35)) * 100

    bmi = weight_kg / ((height_cm / 100) ** 2)

    if bmi < 21:
        fat = np.clip(fat, 8 if gender == "male" else 16, 24)
    else:
        fat = np.clip(fat, 8, 38)

    return float(fat)

# =========================
# FAT BODY PART DETECTION
# =========================
def detect_fat_body_part(widths):
    regions = {
        "Chest": np.mean(widths[0:3]),
        "Upper Abdomen": np.mean(widths[3:6]),
        "Stomach / Waist": np.mean(widths[6:10]),
        "Hips": np.mean(widths[10:13]),
        "Legs": np.mean(widths[13:16])
    }

    dominant_part = max(regions, key=regions.get)
    return dominant_part

# =========================
# CATEGORY
# =========================
def fat_category(fat, gender):
    if gender == "male":
        if fat < 14: return "Athletic"
        if fat < 18: return "Fit"
        if fat < 25: return "Average"
        return "High Body Fat"
    else:
        if fat < 21: return "Athletic"
        if fat < 25: return "Fit"
        if fat < 32: return "Average"
        return "High Body Fat"

# =========================
# CACHING
# =========================
async def get_cached_result(user_id: str):
    cached = await redis_client.get(f"user:{user_id}:body_fat_estimate")
    return eval(cached) if cached else None

async def cache_result(user_id: str, result: dict):
    await redis_client.setex(f"user:{user_id}:body_fat_estimate", 300, str(result))

# =========================
# PARALLEL IMAGE PROCESSING
# =========================
async def process_single_image(img, height_cm, view):
    try:
        image = cv2.imdecode(np.frombuffer(await img.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None

        validate_image(image)
        mask = await segment_body(image)
        px_h, widths = extract_slices(mask)
        volume = estimate_volume(px_h, widths, height_cm, view)
        return volume, widths
    except Exception:
        return None

# =========================
# API
# =========================
@app.post("/estimate-body-fat")
@limiter.limit("5/minute")
async def estimate(
    request: Request,  # Add this line
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    gender: str = Form(...),
    user_id: Optional[str] = Form(None)
):
    if gender not in ["male", "female"]:
        raise HTTPException(400, "gender must be male or female")
    if len(images) < 2:
        raise HTTPException(400, "Front and side images required")

    # Generate a user_id if not provided
    user_id = user_id or str(uuid.uuid4())

    # Check cache
    cached_result = await get_cached_result(user_id)
    if cached_result:
        return cached_result

    # Process images in parallel
    tasks = [
        process_single_image(images[0], height_cm, "front"),
        process_single_image(images[1], height_cm, "side")
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle results
    volumes = []
    all_widths = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception) or result is None:
            continue
        volumes.append(result[0])
        all_widths.append(result[1])

    # Fallback logic
    if not volumes:
        raise HTTPException(400, "Failed to process images")

    avg_volume = float(np.mean(volumes))
    avg_widths = np.mean(np.array(all_widths), axis=0)

    fat = estimate_body_fat(avg_volume, weight_kg, height_cm, gender)
    dominant_body_part = detect_fat_body_part(avg_widths)

    result = {
        "body_fat_estimate": {
            "total_body_fat": f"{round(fat,1)}%",
            "category": fat_category(fat, gender),
            "confidence": "High" if len(volumes) >= 2 else "Medium"
        },
        "fat_mostly_present_in": dominant_body_part,
        "inputs_used": {
            "views": "front + side" if len(volumes) >= 2 else "front only" if volumes else "none",
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "gender": gender
        },
        "method": "Multi-view silhouette volumetric analysis",
        "note": "AI-based visual estimation, non-medical"
    }

    # Cache the result
    background_tasks.add_task(cache_result, user_id, result)

    return result

# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fatpercentage:app", host="127.0.0.1", port=8000)
