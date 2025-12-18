from fastapi import FastAPI, UploadFile, File
import shutil
import os
import redis
import json
import time
from celery import Celery
import logging
from multiprocessing import Process, Queue, Manager

# Import các hàm worker mới từ rtsp_worker
from rtsp_worker import run_inference_server, run_camera_streamer
uvicorn_logger = logging.getLogger("uvicorn.access")

class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Nếu trong log message có chứa '/camera/status/', trả về False để không in log
        return "/camera/status/" not in record.getMessage()

# Thêm filter vào logger
uvicorn_logger.addFilter(HealthCheckFilter())
app = FastAPI(title="MoViNet Video & RTSP System - Shared Model Optimized")

# Kết nối Redis
redis_client = redis.Redis(host='redis_server_ai', port=6379, db=0)
celery_client = Celery('video_tasks', 
                       broker='redis://redis_server_ai:6379/0', 
                       backend='redis://redis_server_ai:6379/0')

# Quản lý tài nguyên dùng chung
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Khởi tạo Queue và Manager để chia sẻ dữ liệu giữa các Process
manager = Manager()
input_queue = Queue(maxsize=150)  # Hàng đợi chứa frame từ các camera
active_cameras = {}              # Lưu các process streamer

@app.on_event("startup")
def startup_event():
    """
    Khi API khởi động, chạy duy nhất 1 Inference Server duy nhất.
    Server này sẽ load model MoViNet vào RAM (tốn ~1.2GB).
    """
    print("--- [SYSTEM] STARTING SHARED INFERENCE SERVER ---")
    p = Process(target=run_inference_server, args=(input_queue,))
    p.daemon = True # Tự động đóng khi app chính đóng
    p.start()

@app.post("/detect_video")
async def detect_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    abs_path = os.path.abspath(file_path)
    task = celery_client.send_task('predict_violence', args=[abs_path])
    return {"task_id": task.id, "message": "Video queued"}

@app.get("/result/{task_id}")
def get_result(task_id: str):
    from celery.result import AsyncResult
    task_result = AsyncResult(task_id, app=celery_client)
    if task_result.state == 'SUCCESS':
        return {"status": "Success", "result": task_result.result}
    elif task_result.state == 'FAILURE':
        return {"status": "Failed", "error": str(task_result.info)}
    else:
        return {"status": task_result.state}

@app.post("/camera/start")
def start_camera(camera_id: str, rtsp_url: str, risk_level: str = "low"):
    """
    Khởi động streamer cho camera. 
    Process này cực nhẹ vì không load Model AI.
    """
    if camera_id in active_cameras:
        if active_cameras[camera_id].is_alive():
            return {"status": "already_running", "camera_id": camera_id}
        else:
            del active_cameras[camera_id]

    redis_client.delete(f"stop_signal_{camera_id}")
    
    # Khởi chạy streamer (Chỉ đọc RTSP và đẩy vào queue)
    print(f"[DEBUG] API: Starting streamer for {camera_id} - Risk: {risk_level}")
    print(f"[DEBUG] Current Queue Size: {input_queue.qsize()}") # Kiểm tra độ đầy của hàng đợi
    p = Process(target=run_camera_streamer, args=(camera_id, rtsp_url, input_queue, risk_level))
    
    p.start()
    
    active_cameras[camera_id] = p
    return {"status": "started", "camera_id": camera_id, "risk": risk_level}

@app.post("/camera/stop")
def stop_camera(camera_id: str):
    if camera_id not in active_cameras:
        return {"status": "error", "message": "Camera not found"}
    
    redis_client.set(f"stop_signal_{camera_id}", "1")
    active_cameras[camera_id].join(timeout=2)
    
    if active_cameras[camera_id].is_alive():
        active_cameras[camera_id].terminate() 
        
    del active_cameras[camera_id]
    return {"status": "stopped"}

@app.get("/camera/status/{camera_id}")
def get_cam_status(camera_id: str):
    """Đọc kết quả từ Redis (do Inference Server ghi vào)"""
    data = redis_client.get(f"cam_status_{camera_id}")
    if not data:
        return {"status": "offline", "message": "No signal from shared server"}
    return json.loads(data)

@app.get("/system/active_cameras")
def list_cameras():
    running = [cid for cid, p in active_cameras.items() if p.is_alive()]
    return {
        "active_cameras": running,
        "queue_size": input_queue.qsize(),
        "mode": "shared_inference_server"
    }