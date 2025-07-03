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
from numpy.linalg import norm
from supabase import create_client
from uuid import uuid4
import os
import base64
import requests
from dotenv import load_dotenv
import face_recognition

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
    """Read image directly from Google Drive without local storage"""
    file_data = drive_service.files().get_media(fileId=file_id).execute()
    # Process directly in memory
    image = np.array(Image.open(io.BytesIO(file_data)))
    return image  # RGB format already

def get_face_encoding(image):
    """Get face encoding using face_recognition library (no local model files)"""
    # Reduce image size if too large to save memory
    h, w = image.shape[:2]
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image = np.array(Image.fromarray(image).resize(new_size))
    
    # Find face locations (much faster with model='hog' and no GPU needed)
    face_locations = face_recognition.face_locations(image, model='hog')
    
    if not face_locations:
        return None
    
    # Get face encodings - these are 128D vectors
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    if not face_encodings:
        return None
    
    return face_encodings[0]  # Return first face encoding

def calculate_similarity(encoding1, encoding2):
    """Calculate similarity between two face encodings"""
    # face_recognition uses distance, smaller is more similar
    return 1 - face_recognition.face_distance([encoding1], encoding2)[0]

# Routes
@app.get("/hello")
async def hello():
    return {"message": "hello"}

@app.get("/health")
async def health_check():
    """Health check endpoint with memory monitoring"""
    try:
        # Using os.getpid() directly instead of psutil
        process_info = os.popen(f"ps -o rss= -p {os.getpid()}").read()
        memory_mb = int(process_info.strip()) / 1024  # Convert KB to MB
        
        return {
            "status": "healthy",
            "memory_usage_mb": round(memory_mb, 1),
            "memory_limit_mb": 512,
            "memory_usage_percent": round((memory_mb / 512) * 100, 1)
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

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
    
    logger.info(f"Processing {len(files)} files for encoding...")
    
    for i, item in enumerate(files):
        try:
            # Log progress
            if i % 10 == 0:
                logger.info(f"Processing file {i+1}/{len(files)}")
            
            # Process image directly in memory - no local storage
            img = read_image_from_drive(item["id"])
            face_encoding = get_face_encoding(img)
            
            if face_encoding is not None:
                encodings.append({
                    "id": item["id"],
                    "name": item["name"],
                    "encoding": face_encoding.tolist()  # Convert to list for JSON serialization
                })
                
            # Clear memory immediately
            del img
            if i % 5 == 0:  # Process in small batches
                pass  # No need for gc.collect(), Python will handle memory
                
        except Exception as e:
            logger.warning(f"Skipping {item['name']}: {e}")
    
    save_encodings(folder_id, encodings)
    
    return {"status": "created", "message": f"Encoding created for {len(encodings)} faces."}

@app.post("/api/match")
async def match_faces(file: UploadFile = File(...), folder_id: str = Form(...)):
    try:
        # Process uploaded image directly in memory
        img_bytes = await file.read()
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

        # Get face encoding
        uploaded_encoding = get_face_encoding(img)
        
        if uploaded_encoding is None:
            return JSONResponse(content={"error": "No face found."}, status_code=400)

        known_data = load_encodings(folder_id)

        if not known_data:
            return JSONResponse(content={"error": "Encodings not found."}, status_code=404)

        matched = []

        async def event_stream():
            total = len(known_data)
            for i, item in enumerate(known_data):
                # Convert back from list to numpy array
                item_encoding = np.array(item["encoding"])
                # Calculate similarity - higher values are better matches
                similarity = calculate_similarity(uploaded_encoding, item_encoding)
                
                if similarity > 0.6:  # Similarity threshold 
                    matched.append({
                        "id": item["id"],
                        "name": item["name"],
                        "similarity": round(similarity * 100, 1),
                        "url": f"https://drive.google.com/uc?export=download&id={item['id']}"
                    })
                yield f"data: {json.dumps({'progress': int((i + 1) / total * 100)})}\n\n"
                time.sleep(0.05)

            # Sort matches by similarity
            matched.sort(key=lambda x: x['similarity'], reverse=True)
            yield f"data: {json.dumps({'progress': 100, 'images': matched})}\n\n"

        # Clear memory after processing
        del img
        result = StreamingResponse(event_stream(), media_type="text/event-stream")
        
        return result

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
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
