from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from googleapiclient.discovery import build
from google.oauth2 import service_account
from PIL import Image
import numpy as np
import io
import pickle
import json
import time
import logging
from deepface import DeepFace
from scipy.spatial.distance import cosine
from supabase import create_client
from uuid import uuid4
import os
import base64
import requests
from dotenv import load_dotenv

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)

# App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load credentials
load_dotenv()
SCOPES = ['https://www.googleapis.com/auth/drive']

encoded_credentials = os.getenv("GOOGLE_SERVICE_ACCOUNT_BASE64")
if not encoded_credentials:
    raise ValueError("Service account Base64 is missing!")
decoded_json = base64.b64decode(encoded_credentials)
credentials = service_account.Credentials.from_service_account_info(
    json.loads(decoded_json), scopes=SCOPES
)
drive_service = build('drive', 'v3', credentials=credentials)

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = "encodings"
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Photo root folder
PHOTOS_FOLDER_ID = os.getenv("PHOTOS_FOLDER_ID")

# Models
class FolderRequest(BaseModel):
    folder_id: str
    force: bool = False

# Utils
def save_encodings(folder_id: str, encodings_data: list):
    buffer = io.BytesIO()
    pickle.dump(encodings_data, buffer)
    buffer.seek(0)
    path = f"{folder_id}.pkl"
    upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/octet-stream",
        "x-upsert": "true"
    }
    response = requests.post(upload_url, headers=headers, data=buffer)
    if response.status_code not in (200, 201):
        raise Exception(f"Upload failed: {response.text}")

def load_encodings(folder_id: str):
    try:
        path = f"{folder_id}.pkl"
        data = supabase.storage.from_(SUPABASE_BUCKET).download(path)
        return pickle.loads(data) if data else None
    except Exception:
        return None

def delete_encoding(folder_id: str):
    path = f"{folder_id}.pkl"
    headers = {"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
    requests.delete(f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}", headers=headers)

def list_drive_files(folder_id: str, mime_type: str = 'image/') -> list:
    query = f"'{folder_id}' in parents and mimeType contains '{mime_type}' and trashed=false"
    response = drive_service.files().list(q=query, fields="files(id, name, webContentLink)").execute()
    return response.get('files', [])

def read_image_from_drive(file_id: str) -> np.ndarray:
    file_data = drive_service.files().get_media(fileId=file_id).execute()
    return np.array(Image.open(io.BytesIO(file_data)))

# Routes
@app.get("/hello")
async def hello():
    return {"message": "hello"}

@app.get("/api/folders")
async def list_folders():
    folders = list_drive_files(PHOTOS_FOLDER_ID, mime_type='application/vnd.google-apps.folder')
    return {"folders": folders}

@app.get("/api/images")
async def list_images(folder_id: str):
    try:
        items = list_drive_files(folder_id)
        images = [{
            "id": item["id"],
            "name": item["name"],
            "url": f"https://drive.google.com/uc?export=download&id={item['id']}"
        } for item in items]
        return {"images": images}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/create_encoding")
async def create_encoding(request: FolderRequest):
    folder_id, force = request.folder_id, request.force
    if load_encodings(folder_id) and not force:
        return {"status": "exists", "message": "Encoding already exists."}
    if force:
        delete_encoding(folder_id)

    files = list_drive_files(folder_id)
    encodings = []
    for item in files:
        try:
            img_np = read_image_from_drive(item["id"])
            result = DeepFace.represent(
                img_path=img_np,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False
            )
            if result:
                encodings.append({
                    "id": item["id"],
                    "name": item["name"],
                    "encoding": result[0]["embedding"]
                })
        except Exception as e:
            logger.warning(f"Skipping {item['name']}: {e}")
    save_encodings(folder_id, encodings)
    return {"status": "created", "message": "Encoding created."}

@app.post("/api/match")
async def match_faces(file: UploadFile = File(...), folder_id: str = Form(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        representation = DeepFace.represent(
            img_path=img_np,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=False
        )

        if not representation:
            return JSONResponse(content={"error": "No face found."}, status_code=400)

        uploaded_embedding = np.array(representation[0]["embedding"])
        known_data = load_encodings(folder_id)

        if not known_data:
            return JSONResponse(content={"error": "Encodings not found."}, status_code=404)

        matched = []

        async def event_stream():
            total = len(known_data)
            for i, item in enumerate(known_data):
                dist = cosine(uploaded_embedding, np.array(item["encoding"]))
                if dist < 0.4:
                    matched.append({
                        "id": item["id"],
                        "name": item["name"],
                        "url": f"https://drive.google.com/uc?export=download&id={item['id']}"
                    })
                yield f"data: {json.dumps({'progress': int((i + 1) / total * 100)})}\n\n"
                time.sleep(0.05)

            yield f"data: {json.dumps({'progress': 100, 'images': matched})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.exception("Match error")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/has-encoding")
async def has_encoding(folder_id: str):
    try:
        files = supabase.storage.from_(SUPABASE_BUCKET).list("")
        exists = any(f["name"] == f"{folder_id}.pkl" for f in files)
        return {"exists": exists}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/file-metadata")
async def file_metadata(file_id: str):
    try:
        data = drive_service.files().get(fileId=file_id, fields="name").execute()
        return {"name": data["name"]}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/file-download")
async def file_download(file_id: str):
    try:
        content = drive_service.files().get_media(fileId=file_id).execute()
        metadata = drive_service.files().get(fileId=file_id, fields="name").execute()
        return StreamingResponse(io.BytesIO(content), media_type="application/octet-stream", headers={
            "Content-Disposition": f'attachment; filename="{metadata.get("name", file_id)}"'
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/delete_encoding")
async def delete_encoding_api(request: FolderRequest):
    try:
        delete_encoding(request.folder_id)
        return {"status": "deleted", "message": "Encoding deleted."}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/check_encoding_exists")
async def check_encoding_exists(request: FolderRequest):
    try:
        exists = load_encodings(request.folder_id) is not None
        return {"exists": exists}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/generate-folder-token")
def generate_folder_token(data: dict):
    token = str(uuid4())
    supabase.table("folder_tokens").insert({
        "folder_name": data["folder_name"],
        "token": token
    }).execute()
    return {"token": token}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("deploy:app", host="0.0.0.0", port=10000)
