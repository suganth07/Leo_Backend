from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from googleapiclient.discovery import build
from google.oauth2 import service_account
from PIL import Image
import face_recognition
import numpy as np
import io
import pickle
import json
import time
import logging
import requests
from supabase import create_client
from uuid import uuid4


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence or suppress httpx info-level logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)



app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'leo-0000.json'
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
drive_service = build('drive', 'v3', credentials=credentials)

PHOTOS_FOLDER_ID = "1yM3_aKiaizjqutcIHtBVzIfdEsy-fouh"
SUPABASE_URL = "https://nisycdwowasgdbwdtxvr.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5pc3ljZHdvd2FzZ2Rid2R0eHZyIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NTgxMzAxNiwiZXhwIjoyMDYxMzg5MDE2fQ.JV02VdUdcmqO4-Lt5dqb44NUJCsz9pwnPyA9jSUA-1o"
SUPABASE_BUCKET = "encodings"
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) 

class FolderRequest(BaseModel):
    folder_id: str
    force: bool = False


def save_encodings(folder_name: str, encodings_data: list):
    buffer = io.BytesIO()
    pickle.dump(encodings_data, buffer)
    buffer.seek(0)

    file_path = f"{folder_name}.pkl"
    upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{file_path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/octet-stream",
        "x-upsert": "true"
    }
    response = requests.post(upload_url, headers=headers, data=buffer)
    print("done")
    if response.status_code not in (200, 201):
        raise Exception(f"Failed to upload encoding: {response.text}")

def load_encodings(folder_name: str):
    try:
        file_path = f"{folder_name}.pkl"
        res = supabase.storage.from_(SUPABASE_BUCKET).download(file_path)
        if not res:
            return None
        return pickle.loads(res)
    except Exception as e:
        logger.warning(f"Failed to load encodings for folder {folder_name}")
        return None

def delete_encoding(folder_name: str):
    delete_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{folder_name}.pkl"
    headers = {"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
    print("deleted")
    requests.delete(delete_url, headers=headers)
    
@app.get("/api/has-encoding")
async def has_encoding(folder_id: str):
    file_path = f"{folder_id}.pkl"

    try:
        files = supabase.storage.from_(SUPABASE_BUCKET).list("")  
        exists = any(f["name"] == file_path for f in files)

        return JSONResponse({"exists": exists}, status_code=200)

    except Exception as e:
        print(f"Error checking encoding: {e}")
        return JSONResponse({"error": "Internal Server Error"}, status_code=500)

def read_image_from_drive(file_id: str) -> np.ndarray:
    file = drive_service.files().get_media(fileId=file_id).execute()
    return np.array(Image.open(io.BytesIO(file)))

def list_drive_files(folder_id: str, mime_type: str = 'image/') -> list:
    query = f"'{folder_id}' in parents and mimeType contains '{mime_type}' and trashed=false"
    results = drive_service.files().list(
        q=query, fields="files(id, name, webContentLink)").execute()
    return results.get('files', [])

@app.get("/api/folders")
async def list_folders():
    try:
        folders = list_drive_files(PHOTOS_FOLDER_ID, mime_type='application/vnd.google-apps.folder')
        return {"folders": folders}
    except Exception as e:
        logger.exception("Error listing folders")
        return JSONResponse(content={"error": str(e)}, status_code=500)
@app.get("/hello")
async def hello():
    return "hello"
@app.get("/api/images")
async def list_images(folder_id: str):
    try:
        items = list_drive_files(folder_id)
        images = []
        for item in items:
            url = item.get('webContentLink')
            if url:
                url = url.replace('&export=download', '') + "&export=download"
            else:
                url = f"https://drive.google.com/uc?export=download&id={item['id']}"
            images.append({
                "id": item['id'],
                "name": item['name'],
                "url": url
            })
        return {"images": images}
    except Exception as e:
        logger.exception("Error listing images")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/create_encoding")
async def check_or_create_encoding(request: FolderRequest):
    folder_id = request.folder_id
    force = request.force
    try:
        encoding_exists = load_encodings(folder_id) is not None

        if encoding_exists and force:
            delete_encoding(folder_id)
            print("Encoding deleted.")

        drive_images = list_drive_files(folder_id)
        encodings_data = []
        for item in drive_images:
            try:
                drive_image = read_image_from_drive(item['id'])
                encs = face_recognition.face_encodings(drive_image, model='cnn')
                if encs:
                    encodings_data.append({
                        "id": item['id'],
                        "name": item['name'],
                        "encoding": encs[0].tolist()
                    })
            except Exception:
                continue

        save_encodings(folder_id, encodings_data)
        print("Encodings saved for folder", folder_id)
        
        return {"status": "created", "message": "Encoding created successfully."}

    except Exception as e:
        logger.exception("Error checking or creating encoding")
        return JSONResponse(content={"error": str(e)}, status_code=500)
@app.post("/api/match")
async def match_faces(file: UploadFile = File(...), folder_id: str = Form(...)):
    try:
        uploaded_image = face_recognition.load_image_file(io.BytesIO(await file.read()))
        uploaded_encodings = face_recognition.face_encodings(uploaded_image, model='cnn')


        if not uploaded_encodings:
            return JSONResponse(content={"error": "No faces found in uploaded image."}, status_code=400)

        uploaded_encoding = uploaded_encodings[0]
        precomputed_data = load_encodings(folder_id)
        if precomputed_data is None:
            return JSONResponse(
                content={"error": "No encoding file found. Please create encodings first."},
                status_code=404
            )

        matched_images = []
        total_images = len(precomputed_data)

        async def event_generator():
            for idx, item in enumerate(precomputed_data):
                known_encoding = np.array(item['encoding'])
                match = face_recognition.compare_faces([known_encoding], uploaded_encoding, tolerance=0.6)

                if match[0]:
                    url = f"https://drive.google.com/uc?export=download&id={item['id']}"
                    matched_images.append({
                        "id": item['id'],
                        "name": item['name'],
                        "url": url
                    })

                progress = int((idx + 1) / total_images * 100)
                yield f"data: {json.dumps({'progress': progress})}\n\n"
                time.sleep(0.05)

            yield f"data: {json.dumps({'progress': 100, 'images': matched_images})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.exception("Error matching faces")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# @app.post("/api/download-zip")
# async def download_zip(request: Request):
#     try:
#         data = await request.json()
#         image_ids = data.get("image_ids", [])

#         if not image_ids:
#             return JSONResponse(content={"error": "No image IDs provided."}, status_code=400)

#         zip_stream = io.BytesIO()
#         with zipfile.ZipFile(zip_stream, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
#             for file_id in image_ids:
#                 try:
#                     file = drive_service.files().get_media(fileId=file_id).execute()
#                     file_metadata = drive_service.files().get(fileId=file_id, fields="name").execute()
#                     filename = file_metadata.get("name", f"{file_id}.jpg")
#                     zf.writestr(filename, file)
#                 except Exception as e:
#                     print(f"Failed to fetch file with ID {file_id}")
#                     continue

#         zip_stream.seek(0)
#         headers = {
#             'Content-Disposition': 'attachment; filename="images.zip"'
#         }
#         return StreamingResponse(zip_stream, media_type="application/zip", headers=headers)

#     except Exception as e:
#         print(f"Error generating ZIP: {str(e)}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/file-metadata")
async def file_metadata(file_id: str):
    try:
        file_metadata = drive_service.files().get(fileId=file_id, fields="name").execute()
        return {"name": file_metadata["name"]}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/file-download")
async def file_download(file_id: str):
    try:
        file = drive_service.files().get_media(fileId=file_id).execute()
        file_metadata = drive_service.files().get(fileId=file_id, fields="name").execute()
        filename = file_metadata.get("name", f"{file_id}.jpg")

        return StreamingResponse(io.BytesIO(file), media_type="application/octet-stream", headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/delete_encoding")
async def delete_encoding_api(request: FolderRequest):
    try:
        delete_encoding(request.folder_id)
        print("Encoding deleted")
        return {"status": "deleted", "message": "Encoding deleted successfully."}
    except Exception as e:
        logger.exception("Error deleting encoding")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/check_encoding_exists")
async def check_encoding_exists(request: FolderRequest):
    try:
        exists = load_encodings(request.folder_id) is not None
        print("Encoding exists:", exists)
        return {"exists": exists}
    except Exception as e:
        logger.exception("Error checking encoding existence")
        # print("Error checking encoding existence:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.post("/generate-folder-token")
def generate_folder_token(data):
    token = str(uuid4())
    # Store token with folder mapping in Supabase
    supabase.table("folder_tokens").insert({
        "folder_name": data.folder_name,
        "token": token
    }).execute()
    return {"token": token}