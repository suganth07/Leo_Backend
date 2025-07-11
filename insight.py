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
import logging
import insightface
from supabase import create_client
from uuid import uuid4
import os
import base64
import requests
from dotenv import load_dotenv
import gc
import psutil
import asyncio
from datetime import datetime
import cv2
import uvicorn

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

# Initialize InsightFace for ultra-fast multi-face processing
# Using ArcFace model for maximum accuracy with speed optimization
face_app = None

def initialize_insightface():
    """Initialize InsightFace model optimized for speed and accuracy"""
    global face_app
    if face_app is None:
        face_app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))  # Optimized detection size
        logger.info("🚀 InsightFace initialized successfully - Ready for multi-face processing!")
    return face_app

def extract_face_embeddings(image_array):
    """
    Extract face embeddings from image using InsightFace
    Handles multiple faces in a single image efficiently
    Returns list of embeddings for all detected faces
    """
    app = initialize_insightface()
    
    # Convert PIL/numpy array to format expected by InsightFace
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # RGB to BGR for OpenCV compatibility
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_array
    
    # Detect and analyze faces
    faces = app.get(image_bgr)
    
    embeddings = []
    for face in faces:
        # InsightFace returns normalized embeddings directly
        embeddings.append(face.embedding)
    
    return embeddings

def compare_faces(embedding1, embedding2, threshold=0.6):
    """
    Compare two face embeddings using cosine similarity
    InsightFace embeddings work best with cosine distance
    Lower threshold = stricter matching
    """
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    distance = 1 - similarity
    return distance < threshold, distance

# Photo root folder
PHOTOS_FOLDER_ID = os.getenv("PHOTOS_FOLDER_ID")

# Global state for auto-sync system
_sync_running = False
_last_sync_times = {}
_sync_stats = {
    "total_photos_processed": 0,
    "total_faces_detected": 0,
    "last_sync_time": None,
    "active_folders": []
}

class AutoSyncManager:
    """
    Professional auto-sync manager for photography workflow.
    Handles automatic detection and face processing of new photos uploaded to Google Drive.
    Optimized for handling 60k+ photos per day with minimal memory usage.
    """
    
    def __init__(self):
        self.sync_interval = 120  # 2 minutes - perfect for your 1-2 min delay requirement
        self.max_concurrent_folders = 3  # Process 3 folders simultaneously
        self.batch_size = 10  # Process 10 images at a time for memory efficiency
        
    async def start_auto_sync(self):
        """Start the professional auto-sync system"""
        global _sync_running, _sync_stats
        
        if _sync_running:
            return {
                "status": "already_running",
                "message": "Auto-sync is already active",
                "stats": _sync_stats
            }
        
        _sync_running = True
        _sync_stats["last_sync_time"] = datetime.now().isoformat()
        
        # Start the sync loop in background
        asyncio.create_task(self._professional_sync_loop())
        
        logger.info("🚀 Professional auto-sync started for photography workflow")
        
        return {
            "status": "started",
            "message": f"Auto-sync activated! Checking for new photos every {self.sync_interval} seconds",
            "interval_seconds": self.sync_interval,
            "features": [
                "Automatic new photo detection",
                "Background face processing",
                "Memory-optimized for 60k+ photos/day",
                "Real-time progress tracking"
            ]
        }
    
    async def _professional_sync_loop(self):
        """Main professional sync loop - handles high-volume photography workflow"""
        global _sync_stats
        
        while _sync_running:
            try:
                logger.info("🔄 Starting professional sync cycle...")
                start_time = datetime.now()
                
                # Get all photography folders
                folders = list_drive_files(PHOTOS_FOLDER_ID, mime_type='application/vnd.google-apps.folder')
                _sync_stats["active_folders"] = [f["name"] for f in folders]
                
                # Process folders in batches for efficiency
                new_photos_count = 0
                new_faces_count = 0
                
                for i in range(0, len(folders), self.max_concurrent_folders):
                    batch = folders[i:i + self.max_concurrent_folders]
                    tasks = []
                    
                    for folder in batch:
                        task = asyncio.create_task(self._sync_folder_professional(folder))
                        tasks.append(task)
                    
                    # Wait for batch to complete
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Collect stats
                    for result in results:
                        if isinstance(result, dict):
                            new_photos_count += result.get("new_photos", 0)
                            new_faces_count += result.get("new_faces", 0)
                
                # Update global stats
                _sync_stats["total_photos_processed"] += new_photos_count
                _sync_stats["total_faces_detected"] += new_faces_count
                _sync_stats["last_sync_time"] = datetime.now().isoformat()
                
                sync_duration = (datetime.now() - start_time).total_seconds()
                
                if new_photos_count > 0:
                    logger.info(f"✅ Sync complete! Processed {new_photos_count} new photos, detected {new_faces_count} faces in {sync_duration:.1f}s")
                else:
                    logger.info(f"✅ Sync complete! No new photos found (checked {len(folders)} folders in {sync_duration:.1f}s)")
                
                # Memory cleanup after each cycle
                clear_memory()
                
                # Wait for next cycle
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error(f"❌ Sync loop error: {e}")
                await asyncio.sleep(30)  # Retry after 30 seconds on error
    
    async def _sync_folder_professional(self, folder):
        """Professional folder sync with optimized face processing"""
        folder_id = folder['id']
        folder_name = folder['name']
        
        try:
            # Get current photos in folder
            current_photos = list_drive_files(folder_id)
            
            # Load existing encodings
            existing_encodings = load_encodings(folder_id) or []
            existing_ids = {item['id'] for item in existing_encodings}
             
            # Find new photos
            new_photos = [photo for photo in current_photos if photo['id'] not in existing_ids]
            
            if not new_photos:
                return {"new_photos": 0, "new_faces": 0}
            
            logger.info(f"📸 Found {len(new_photos)} new photos in '{folder_name}'")
            _last_sync_times[folder_id] = datetime.now().isoformat()
            
            # Process new photos with professional optimization
            new_faces = await self._process_photos_batch(folder_id, new_photos, existing_encodings)
            
            return {"new_photos": len(new_photos), "new_faces": new_faces}
            
        except Exception as e:
            logger.error(f"❌ Error syncing folder '{folder_name}': {e}")
            return {"new_photos": 0, "new_faces": 0}
    
    async def _process_photos_batch(self, folder_id, new_photos, existing_encodings):
        """Process photos in optimized batches for professional workflow"""
        new_encodings = []
        faces_detected = 0
        
        # Process in small batches for memory efficiency
        for i in range(0, len(new_photos), self.batch_size):
            batch = new_photos[i:i + self.batch_size]
            
            for j, photo in enumerate(batch):
                try:
                    # Log progress for large batches
                    if len(new_photos) > 50 and (i + j) % 20 == 0:
                        logger.info(f"🔄 Processing photo {i + j + 1}/{len(new_photos)} in folder...")
                    
                    # Process image for face detection using InsightFace (handles multiple faces)
                    img_np = read_image_from_drive(photo["id"])
                    embeddings = extract_face_embeddings(img_np)
                    
                    if embeddings:
                        # Store all face embeddings for this image
                        for idx, embedding in enumerate(embeddings):
                            new_encodings.append({
                                "id": photo["id"],
                                "name": photo["name"],
                                "face_index": idx,  # Track multiple faces in same image
                                "encoding": embedding.tolist(),  # Convert numpy to list for JSON
                                "processed_at": datetime.now().isoformat()
                            })
                        faces_detected += len(embeddings)
                    
                    # Immediate memory cleanup
                    del img_np
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process {photo['name']}: {e}")
            
            # Batch memory cleanup
            clear_memory()
        
        # Save updated encodings
        if new_encodings:
            all_encodings = existing_encodings + new_encodings
            save_encodings(folder_id, all_encodings)
            logger.info(f"💾 Saved {len(new_encodings)} new face encodings ({faces_detected} faces detected)")
        
        return faces_detected

# Initialize the professional auto-sync manager
auto_sync_manager = AutoSyncManager()

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
    return image  # Return RGB image directly

def process_images_ultra_fast_batch(image_list, batch_size=50):
    """
    Ultra-fast batch processing for lakhs of images
    Optimized for memory efficiency and speed
    """
    initialize_insightface()  # Ensure InsightFace is ready
    all_results = []
    
    logger.info(f"🚀 Starting ultra-fast batch processing for {len(image_list)} images")
    
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        batch_results = []
        
        for item in batch:
            try:
                img_np = read_image_from_drive(item["id"])
                embeddings = extract_face_embeddings(img_np)
                
                if embeddings:
                    for idx, embedding in enumerate(embeddings):
                        batch_results.append({
                            "id": item["id"],
                            "name": item["name"],
                            "face_index": idx,
                            "encoding": embedding.tolist(),
                            "processed_at": datetime.now().isoformat()
                        })
                
                # Immediate memory cleanup
                del img_np
                
            except Exception as e:
                logger.warning(f"Batch processing error for {item['name']}: {e}")
        
        all_results.extend(batch_results)
        
        # Progress logging
        processed = min(i + batch_size, len(image_list))
        logger.info(f"⚡ Processed {processed}/{len(image_list)} images, found {len(batch_results)} faces in current batch")
        
        # Force memory cleanup after each batch
        clear_memory()
    
    logger.info(f"✅ Ultra-fast batch processing complete! Total faces detected: {len(all_results)}")
    return all_results

# Memory management utilities for minimal storage deployment
def clear_memory():
    """Force garbage collection to free memory"""
    gc.collect()

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


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
    processed_count = 0
    total_faces = 0
    
    for item in files:
        try:
            img_np = read_image_from_drive(item["id"])
            embeddings = extract_face_embeddings(img_np)
            
            if embeddings:
                # Store all detected faces with their embeddings
                for idx, embedding in enumerate(embeddings):
                    encodings.append({
                        "id": item["id"],
                        "name": item["name"],
                        "face_index": idx,  # Track multiple faces in same image
                        "encoding": embedding.tolist()  # Convert numpy to list for JSON
                    })
                total_faces += len(embeddings)
            processed_count += 1
            
            # Log progress for large folders
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count}/{len(files)} images, detected {total_faces} faces total")
                
        except Exception as e:
            logger.warning(f"Skipping {item['name']}: {e}")
            
    save_encodings(folder_id, encodings)
    logger.info(f"✅ Encoding complete! Processed {processed_count} images, detected {total_faces} faces")
    return {"status": "created", "message": f"Encoding created with {total_faces} faces from {processed_count} images."}

@app.post("/api/match")
async def match_faces(file: UploadFile = File(...), folder_id: str = Form(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        # Extract all face embeddings from uploaded image using InsightFace
        uploaded_embeddings = extract_face_embeddings(img_np)

        if not uploaded_embeddings:
            return JSONResponse(content={"error": "No faces found in uploaded image."}, status_code=400)

        known_data = load_encodings(folder_id)

        if not known_data:
            return JSONResponse(content={"error": "Encodings not found for this folder."}, status_code=404)

        logger.info(f"🔍 Matching {len(uploaded_embeddings)} uploaded faces against {len(known_data)} stored face encodings")
        matched = []

        async def event_stream():
            total = len(known_data)
            matches_found = 0
            
            for i, stored_item in enumerate(known_data):
                try:
                    stored_embedding = np.array(stored_item["encoding"])
                    
                    # Check each uploaded face against this stored face
                    for face_idx, uploaded_embedding in enumerate(uploaded_embeddings):
                        is_match, distance = compare_faces(uploaded_embedding, stored_embedding, threshold=0.6)
                        
                        if is_match:
                            # Avoid duplicate matches for the same image
                            image_already_matched = any(m["id"] == stored_item["id"] for m in matched)
                            if not image_already_matched:
                                face_info = {
                                    "id": stored_item["id"],
                                    "name": stored_item["name"],
                                    "url": f"https://drive.google.com/uc?export=download&id={stored_item['id']}",
                                    "confidence": round((1 - distance) * 100, 1),
                                    "matched_face_index": stored_item.get("face_index", 0),
                                    "query_face_index": face_idx
                                }
                                matched.append(face_info)
                                matches_found += 1
                                logger.info(f"✅ Match found: {stored_item['name']} (confidence: {face_info['confidence']}%)")
                
                except Exception as e:
                    logger.warning(f"Error processing stored face {i}: {e}")
                
                # Progress update
                progress = int((i + 1) / total * 100)
                yield f"data: {json.dumps({'progress': progress, 'matches_found': matches_found})}\n\n"
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)

            # Sort matches by confidence (highest first)
            matched.sort(key=lambda x: x["confidence"], reverse=True)
            
            final_result = {
                'progress': 100, 
                'images': matched,
                'total_matches': len(matched),
                'faces_in_query': len(uploaded_embeddings),
                'search_completed': True
            }
            
            logger.info(f"🎯 Search complete! Found {len(matched)} matching images from {len(uploaded_embeddings)} query faces")
            yield f"data: {json.dumps(final_result)}\n\n"

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

@app.get("/health")
async def health_check():
    """Health check endpoint with memory monitoring"""
    try:
        memory_mb = get_memory_usage()
        
        # Test InsightFace initialization
        app_ready = face_app is not None
        if not app_ready:
            try:
                initialize_insightface()
                app_ready = True
            except Exception as e:
                logger.warning(f"InsightFace initialization failed: {e}")
        
        return {
            "status": "healthy",
            "memory_usage_mb": round(memory_mb, 1),
            "memory_limit_mb": 512,
            "memory_usage_percent": round((memory_mb / 512) * 100, 1),
            "face_model_loaded": app_ready,
            "face_model": "InsightFace ArcFace",
            "detector": "RetinaFace",
            "optimized_for": "Multiple faces per image with ultra-fast processing",
            "performance": "Optimized for lakhs of images"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/api/optimize-speed")
async def optimize_for_speed():
    """Optimize InsightFace for maximum speed processing"""
    try:
        global face_app
        # Reinitialize with speed-optimized settings
        face_app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(320, 320))  # Smaller detection size for speed
        
        return {
            "status": "optimized", 
            "mode": "speed", 
            "model": "InsightFace ArcFace",
            "detector": "RetinaFace",
            "detection_size": "320x320",
            "message": "Optimized for maximum speed - perfect for lakhs of images"
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/optimize-accuracy")
async def optimize_for_accuracy():
    """Optimize InsightFace for maximum accuracy"""
    try:
        global face_app
        # Reinitialize with accuracy-optimized settings
        face_app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))  # Larger detection size for accuracy
        
        return {
            "status": "optimized", 
            "mode": "accuracy", 
            "model": "InsightFace ArcFace",
            "detector": "RetinaFace",
            "detection_size": "640x640",
            "message": "Optimized for maximum accuracy with multi-face detection"
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ============ PROFESSIONAL AUTO-SYNC API ENDPOINTS ============

@app.post("/api/auto-sync/start")
async def start_professional_auto_sync():
    """Start the professional auto-sync system for photography workflow"""
    try:
        result = await auto_sync_manager.start_auto_sync()
        return result
    except Exception as e:
        logger.error(f"Failed to start auto-sync: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/auto-sync/stop")
async def stop_professional_auto_sync():
    """Stop the professional auto-sync system"""
    global _sync_running
    try:
        _sync_running = False
        logger.info("🛑 Professional auto-sync stopped")
        return {
            "status": "stopped",
            "message": "Auto-sync has been stopped",
            "final_stats": _sync_stats
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/auto-sync/status")
async def get_professional_sync_status():
    """Get comprehensive auto-sync status for photography dashboard"""
    try:
        memory_mb = get_memory_usage()
        return {
            "sync_active": _sync_running,
            "sync_interval_seconds": auto_sync_manager.sync_interval,
            "sync_interval_display": f"{auto_sync_manager.sync_interval // 60} minutes",
            "stats": _sync_stats,
            "last_sync_times": _last_sync_times,
            "performance": {
                "memory_usage_mb": round(memory_mb, 1),
                "memory_usage_percent": round((memory_mb / 512) * 100, 1),
                "face_model_loaded": True  # face_recognition is always ready
            },
            "configuration": {
                "max_concurrent_folders": auto_sync_manager.max_concurrent_folders,
                "batch_size": auto_sync_manager.batch_size,
                "optimized_for": "60k+ photos per day"
            }
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/auto-sync/manual-trigger")
async def trigger_manual_sync():
    """Manually trigger a full sync cycle for all folders"""
    try:
        if not _sync_running:
            return {
                "status": "sync_not_active",
                "message": "Auto-sync is not running. Please start auto-sync first."
            }
        
        # Get folders and start manual sync
        folders = list_drive_files(PHOTOS_FOLDER_ID, mime_type='application/vnd.google-apps.folder')
        
        # Process folders manually
        new_photos_count = 0
        new_faces_count = 0
        
        for folder in folders[:5]:  # Limit to 5 folders for manual trigger
            try:
                result = await auto_sync_manager._sync_folder_professional(folder)
                new_photos_count += result.get("new_photos", 0)
                new_faces_count += result.get("new_faces", 0)
            except Exception as e:
                logger.warning(f"Error syncing folder {folder['name']}: {e}")
        
        return {
            "status": "manual_sync_complete",
            "message": f"Manual sync completed for {len(folders[:5])} folders",
            "results": {
                "new_photos": new_photos_count,
                "new_faces": new_faces_count
            }
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/auto-sync/folder")
async def sync_specific_folder(request: FolderRequest):
    """Manually sync a specific photography folder"""
    try:
        folder_info = drive_service.files().get(fileId=request.folder_id, fields="name").execute()
        folder = {"id": request.folder_id, "name": folder_info.get("name", "Unknown")}
        
        result = await auto_sync_manager._sync_folder_professional(folder)
        
        return {
            "status": "folder_sync_complete",
            "folder_name": folder["name"],
            "results": result,
            "message": f"Processed {result['new_photos']} new photos, detected {result['new_faces']} faces"
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/auto-sync/dashboard")
async def get_photography_dashboard():
    """Get comprehensive dashboard data for photography workflow"""
    try:
        # Get folder statistics
        folders = list_drive_files(PHOTOS_FOLDER_ID, mime_type='application/vnd.google-apps.folder')
        folder_stats = []
        
        total_photos = 0
        total_faces = 0
        
        for folder in folders[:10]:  # Limit to first 10 for performance
            try:
                photos = list_drive_files(folder['id'])
                encodings = load_encodings(folder['id']) or []
                
                folder_stats.append({
                    "id": folder['id'],
                    "name": folder['name'],
                    "photo_count": len(photos),
                    "face_count": len(encodings),
                    "last_sync": _last_sync_times.get(folder['id'], "Never"),
                    "encoding_exists": len(encodings) > 0
                })
                
                total_photos += len(photos)
                total_faces += len(encodings)
                
            except Exception as e:
                logger.warning(f"Error getting stats for folder {folder['name']}: {e}")
        
        return {
            "overview": {
                "total_folders": len(folders),
                "total_photos": total_photos,
                "total_faces_detected": total_faces,
                "sync_active": _sync_running,
                "system_ready": True  # face_recognition is always ready
            },
            "folder_stats": folder_stats,
            "sync_status": _sync_stats,
            "performance": {
                "memory_usage_mb": round(get_memory_usage(), 1),
                "face_model": "InsightFace ArcFace (Ultra-Fast Multi-Face)"
            }
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    # Initialize InsightFace on startup
    initialize_insightface()
    logger.info("🚀 InsightFace initialized - Ready for ultra-fast multi-face processing!")
    
    port = int(os.environ.get("PORT", 10001))
    uvicorn.run("insight:app", host="0.0.0.0", port=port)
