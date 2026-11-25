from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from passlib.context import CryptContext
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError
from pydantic import BaseModel, Field

load_dotenv()

ROOT = Path(__file__).resolve().parent
ENGINE_DIR = ROOT / "engine"
sys.path.insert(0, str(ENGINE_DIR))

try:
    import torch
    import radiology_ai_assistant as engine
except Exception as exc:
    torch = None
    engine = None
    ENGINE_IMPORT_ERROR = str(exc)
else:
    ENGINE_IMPORT_ERROR = None

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "radiology_ai_assistant")
JWT_SECRET = os.getenv("JWT_SECRET", "development-only-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))
WORKDIR = Path(os.getenv("WORKDIR", "./workdir"))
PRESET = os.getenv("MODEL_PRESET", "clahe_aug")
DEVICE_NAME = os.getenv("MODEL_DEVICE", "auto")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
app = FastAPI(title="Radiology AI Assistant API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://radiology-ai-frontend.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mongo_client = None
users = None
scans = None
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2500)
    mongo_client.admin.command("ping")
    db = mongo_client[DB_NAME]
    users, scans = db["users"], db["scans"]
    users.create_index("email", unique=True)
    scans.create_index([("user_id", 1), ("created_at", DESCENDING)])
except Exception as exc:
    print(f"[WARN] MongoDB unavailable: {exc}. Auth/history endpoints will be unavailable.")

inference = None
if engine is not None and torch is not None:
    try:
        inference = engine.InferenceService(
            WORKDIR, PRESET, engine.pick_device(DEVICE_NAME)
        )
    except Exception as exc:
        print(f"[WARN] Could not initialize inference service: {exc}")


class AuthBody(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)


def require_db():
    if users is None or scans is None:
        raise HTTPException(503, "MongoDB is unavailable. Check MONGO_URI and start MongoDB.")


def create_token(user_id: str, email: str):
    expires = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    return jwt.encode({"sub": user_id, "email": email, "exp": expires}, JWT_SECRET, algorithm=JWT_ALGORITHM)


def current_user(authorization: Optional[str] = Header(default=None)):
    require_db()
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token")
    except JWTError:
        raise HTTPException(401, "Invalid or expired token")
    user = users.find_one({"_id": user_id}, {"password_hash": 0})
    if not user:
        raise HTTPException(401, "User no longer exists")
    return user


@app.get("/")
def root():
    return {"message": "Radiology AI Assistant API", "docs": "/docs"}


@app.get("/api/health")
def health():
    branches = inference.branches if inference else []
    return {
        "status": "ok",
        "engine_imported": engine is not None,
        "engine_import_error": ENGINE_IMPORT_ERROR,
        "model_preset": PRESET,
        "available_branches": branches,
        "classes": engine.CLASSES if engine else ["Normal", "COVID-19", "Pneumonia", "Tuberculosis"],
        "database_connected": users is not None,
    }


@app.post("/api/auth/register")
def register(body: AuthBody):
    require_db()
    email = body.email.strip().lower()
    if users.find_one({"email": email}):
        raise HTTPException(409, "An account with this email already exists")
    user_id = str(uuid.uuid4())
    users.insert_one({
        "_id": user_id,
        "email": email,
        "password_hash": pwd_context.hash(body.password),
        "created_at": datetime.now(timezone.utc),
    })
    return {"token": create_token(user_id, email), "user": {"id": user_id, "email": email}}


@app.post("/api/auth/login")
def login(body: AuthBody):
    require_db()
    email = body.email.strip().lower()
    user = users.find_one({"email": email})
    if not user or not pwd_context.verify(body.password, user["password_hash"]):
        raise HTTPException(401, "Incorrect email or password")
    return {"token": create_token(user["_id"], email), "user": {"id": user["_id"], "email": email}}


@app.get("/api/auth/me")
def me(user=__import__("fastapi").Depends(current_user)):
    return {"id": user["_id"], "email": user["email"]}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    branch: str = Form("cnn"),
    user=__import__("fastapi").Depends(current_user),
):
    if inference is None:
        raise HTTPException(503, "AI engine could not be loaded. Check backend/engine and dependencies.")
    if branch not in inference.branches:
        raise HTTPException(503, f"Model branch '{branch}' is not trained or available. Available: {inference.branches}")
    data = await file.read(engine.MAX_UPLOAD_BYTES + 1)
    try:
        result = inference.predict(file.filename or "upload.png", data, branch)
    except engine.InferenceError as exc:
        raise HTTPException(exc.status, exc.msg)
    scan_id = str(uuid.uuid4())
    record = {
        "_id": scan_id,
        "user_id": user["_id"],
        "filename": file.filename or "upload.png",
        "predicted_class": result["predicted_class"],
        "branch": branch,
        "preset": result.get("preset"),
        "latency_ms": result.get("latency_ms"),
        "created_at": datetime.now(timezone.utc),
        "result": result,
    }
    try:
        scans.insert_one(record)
    except Exception as exc:
        raise HTTPException(503, f"Prediction completed but scan history could not be saved: {exc}")
    result["scan_id"] = scan_id
    return result


@app.get("/api/history")
def history(user=__import__("fastapi").Depends(current_user), limit: int = 30):
    require_db()
    limit = max(1, min(limit, 100))
    docs = scans.find({"user_id": user["_id"]}).sort("created_at", DESCENDING).limit(limit)
    return [{
        "id": d["_id"],
        "filename": d["filename"],
        "predicted_class": d["predicted_class"],
        "branch": d["branch"],
        "created_at": d["created_at"].isoformat(),
        "latency_ms": d.get("latency_ms"),
    } for d in docs]


@app.get("/api/history/{scan_id}")
def history_detail(scan_id: str, user=__import__("fastapi").Depends(current_user)):
    require_db()
    doc = scans.find_one({"_id": scan_id, "user_id": user["_id"]})
    if not doc:
        raise HTTPException(404, "Scan not found")
    result = doc["result"]
    # Do not return any stored source image; only result metadata and optional Grad-CAM.
    return result


@app.exception_handler(PyMongoError)
async def mongo_error_handler(_, exc):
    return JSONResponse(status_code=503, content={"detail": "Database operation failed"})
