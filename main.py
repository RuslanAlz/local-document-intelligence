import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from pydantic import BaseModel
from pipeline import process_document

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Extraction API", description="Local AI Document Summarizer & Extractor")

jobs = {}

class JobStatus(BaseModel):
    job_id: str
    status: str
    result: str = None
    error: str = None

def run_pipeline_task(job_id: str, file_path: str, force_restart: bool = False):
    jobs[job_id] = {"status": "processing", "result": None, "error": None}
    try:
        result = process_document(file_path, force_restart=force_restart)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result
    except Exception as e:
        logger.exception("Pipeline execution failed for job %s: %s", job_id, e)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, force_restart: bool = False, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.gettempdir())
    file_path = str(temp_dir / f"{job_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    background_tasks.add_task(run_pipeline_task, job_id, file_path, force_restart)
    return {"job_id": job_id, "message": "Document processing started"}

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    if job_id not in jobs:
        return {"job_id": job_id, "status": "not_found"}
    return {"job_id": job_id, **jobs[job_id]}
