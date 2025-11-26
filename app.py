# app.py
from fastapi import FastAPI, UploadFile, File
import shutil
import os
from worker import predict_violence # Import task từ worker
from celery.result import AsyncResult

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/detect")
async def detect_violence(file: UploadFile = File(...)):
    # 1. Lưu file video xuống đĩa
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Gửi task cho Worker (Không chờ kết quả ngay)
    task = predict_violence.delay(file_location)
    
    # 3. Trả về Task ID để client tra cứu sau
    return {"task_id": task.id, "message": "Video đang được xử lý"}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    task_result = AsyncResult(task_id)
    if task_result.state == 'PENDING':
        return {"status": "Pending..."}
    elif task_result.state == 'PROCESSING':
        return {"status": "Processing..."}
    elif task_result.state == 'SUCCESS':
        return {"status": "Success", "result": task_result.result}
    else:
        return {"status": "Failed", "error": str(task_result.info)}
