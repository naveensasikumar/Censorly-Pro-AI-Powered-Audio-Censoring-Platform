from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import os
import shutil
import uuid
import zipfile
from typing import List
import asyncio
from datetime import datetime

# Import your Censorly system
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from censorly_system.audio_censoring import AudioCensoringSystem
from censorly_system.text_detector import AdvancedOffensiveTextDetector

app = FastAPI(title="Censorly API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Initialize Censorly System
try:
    audio_censor = AudioCensoringSystem("censorly_system/advanced_offensive_detector.pkl")
    print("Censorly system loaded successfully")
except Exception as e:
    print(f"Error loading Censorly system: {e}")
    audio_censor = None

# Job tracking
processing_jobs = {}

@app.get("/")
async def root():
    return {
        "message": "Censorly API is running",
        "censorly_loaded": audio_censor is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    severity: str = Form("teen"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload files and start background processing"""
    
    if not audio_censor:
        raise HTTPException(status_code=503, detail="Censorly system not available")
    
    if len(files) > 10:  # Limit number of files
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded files
    uploaded_files = []
    for file in files:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")
        
        # Save file
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        saved_filename = f"{file_id}{file_extension}"
        file_path = f"uploads/{saved_filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        uploaded_files.append({
            "file_id": file_id,
            "original_name": file.filename,
            "saved_path": file_path,
            "size": os.path.getsize(file_path),
            "status": "queued"
        })
    
    # Initialize job tracking
    processing_jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "total_files": len(uploaded_files),
        "completed_files": 0,
        "files": uploaded_files,
        "severity": severity,
        "created_at": datetime.now().isoformat(),
        "download_ready": False,
        "error": None
    }
    
    # Start background processing
    background_tasks.add_task(process_audio_files, job_id, uploaded_files, severity)
    
    return {
        "job_id": job_id,
        "files_uploaded": len(uploaded_files),
        "status": "processing_started",
        "message": f"Processing {len(uploaded_files)} files with {severity} severity level"
    }

async def process_audio_files(job_id: str, files: List[dict], severity: str):
    """Background task to process audio files with Censorly"""
    
    # Map frontend severity to Censorly severity
    severity_mapping = {
        'children': 'LOW',     
        'teen': 'MEDIUM',       
        'adult': 'CRITICAL'    
    }
    
    censorly_severity = severity_mapping.get(severity, 'MEDIUM')
    processed_files = []
    
    try:
        for i, file_info in enumerate(files):
            # Update job status
            processing_jobs[job_id]["files"][i]["status"] = "processing"
            
            try:
                # Process with Censorly
                output_filename = f"censored_{file_info['file_id']}.mp3"
                output_path = f"outputs/{output_filename}"
                
                result = audio_censor.process_audio_file(
                    input_audio_path=file_info['saved_path'],
                    output_audio_path=output_path,
                    severity_filter=censorly_severity,
                    replacement_type="beep"
                )
                
                # Update file info with results
                file_info.update({
                    "status": "completed",
                    "output_path": output_path,
                    "output_filename": output_filename,
                    "words_found": result['total_offensive_found'],
                    "words_censored": result['censored_count'],
                    "transcript": result['transcription'][:200] + "..." if len(result['transcription']) > 200 else result['transcription'],
                    "processing_time": result.get('processing_time', 0)
                })
                
                processed_files.append(file_info)
                
                # Update job progress
                processing_jobs[job_id]["completed_files"] += 1
                processing_jobs[job_id]["files"][i] = file_info
                
                print(f"Processed {file_info['original_name']}: {result['censored_count']} words censored")
                
            except Exception as e:
                # Handle individual file errors
                file_info.update({
                    "status": "error",
                    "error": str(e)
                })
                processing_jobs[job_id]["files"][i] = file_info
                print(f"Error processing {file_info['original_name']}: {e}")
        
        # Create ZIP file if multiple files
        if len(processed_files) > 1:
            zip_path = f"outputs/{job_id}_censored_batch.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file_info in processed_files:
                    if file_info.get("output_path") and os.path.exists(file_info["output_path"]):
                        zipf.write(
                            file_info["output_path"], 
                            f"censored_{file_info['original_name']}"
                        )
            
            processing_jobs[job_id]["zip_file"] = zip_path
        
        # Mark job as completed
        processing_jobs[job_id].update({
            "status": "completed",
            "download_ready": True,
            "completed_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        # Handle job-level errors
        processing_jobs[job_id].update({
            "status": "error",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })
        print(f"Job {job_id} failed: {e}")

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get processing job status"""
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    # Calculate progress percentage
    progress = 0
    if job["total_files"] > 0:
        progress = (job["completed_files"] / job["total_files"]) * 100
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": round(progress, 1),
        "total_files": job["total_files"],
        "completed_files": job["completed_files"],
        "download_ready": job.get("download_ready", False),
        "error": job.get("error"),
        "files": job["files"]
    }

@app.get("/api/download/{job_id}")
async def download_results(job_id: str):
    """Get download links for processed files"""
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if not job.get("download_ready"):
        raise HTTPException(status_code=400, detail="Processing not completed")
    
    # Prepare download links
    download_links = []
    
    for file_info in job["files"]:
        if file_info.get("status") == "completed" and file_info.get("output_filename"):
            download_links.append({
                "original_name": file_info["original_name"],
                "download_url": f"/api/file/{file_info['output_filename']}",
                "words_censored": file_info.get("words_censored", 0),
                "transcript_preview": file_info.get("transcript", "")
            })
    
    response = {
        "job_id": job_id,
        "files": download_links,
        "total_files": len(download_links)
    }
    
    # Add ZIP download if multiple files
    if job.get("zip_file") and len(download_links) > 1:
        zip_filename = os.path.basename(job["zip_file"])
        response["zip_download"] = f"/api/file/{zip_filename}"
    
    return response

@app.get("/api/file/{filename}")
async def download_file(filename: str):
    """Download individual processed file"""
    
    file_path = f"outputs/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path, 
        filename=filename,
        media_type='application/octet-stream'
    )

@app.delete("/api/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up job files (optional endpoint)"""
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    # Delete uploaded files
    for file_info in job["files"]:
        if os.path.exists(file_info["saved_path"]):
            os.remove(file_info["saved_path"])
        
        if file_info.get("output_path") and os.path.exists(file_info["output_path"]):
            os.remove(file_info["output_path"])
    
   
    if job.get("zip_file") and os.path.exists(job["zip_file"]):
        os.remove(job["zip_file"])
    
    
    del processing_jobs[job_id]
    
    return {"message": "Job cleaned up successfully"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)