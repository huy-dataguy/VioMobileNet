from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import redis
import json
import time
from celery import Celery
import logging
from multiprocessing import Process, Queue, Manager

# Import các hàm worker từ rtsp_worker
from rtsp_worker import run_inference_server, run_camera_streamer

# Cấu hình logging để lọc bỏ các bản tin health check status của camera trên UI
uvicorn_logger = logging.getLogger("uvicorn.access")
class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/camera/status/" not in record.getMessage()

uvicorn_logger.addFilter(HealthCheckFilter())

app = FastAPI(title="VioMobileNet - Dual Inference Pipeline Optimized")

# Cấu hình CORS cho phép UI truy vấn API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kết nối Redis dùng chung
redis_client = redis.Redis(host='redis_server_ai', port=6379, db=0)
celery_client = Celery('video_tasks', 
                       broker='redis://redis_server_ai:6379/0', 
                       backend='redis://redis_server_ai:6379/0')

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- KHỞI TẠO HỆ THỐNG QUEUE CHO DUAL PIPELINE ---
# Queue cho camera High Risk (Ưu tiên cao, kích thước nhỏ để đảm bảo Real-time)
high_risk_queue = Queue(maxsize=30) 
# Queue cho camera Medium/Low Risk (Tải chung, kích thước lớn hơn một chút)
low_med_queue = Queue(maxsize=50)   

active_cameras = {} # Lưu trữ các process streamer đang chạy

@app.on_event("startup")
def startup_event():
    """
    Khi API khởi động, chạy 2 Inference Servers song song trên 2 process riêng biệt.
    Mỗi Server nạp 1 bản Model MoViNet vào RAM (~1.5GB/instance).
    """
    print("--- [SYSTEM] STARTING DUAL INFERENCE PIPELINE ---")
    
    # Luồng 1: Chuyên xử lý các camera khu vực High Risk
    p_high = Process(target=run_inference_server, args=(high_risk_queue, "HIGH_SERVER"))
    p_high.daemon = True
    p_high.start()

    # Luồng 2: Xử lý các camera khu vực Medium và Low Risk
    p_low_med = Process(target=run_inference_server, args=(low_med_queue, "LOW_MED_SERVER"))
    p_low_med.daemon = True
    p_low_med.start()

@app.post("/detect_video")
async def detect_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    abs_path = os.path.abspath(file_path)
    task = celery_client.send_task('predict_violence', args=[abs_path])
    return {"task_id": task.id, "message": "Video queued for offline processing"}

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
    Khởi động streamer cho camera và điều hướng vào đúng Queue dựa trên Risk Level.
    """
    if camera_id in active_cameras:
        if active_cameras[camera_id].is_alive():
            return {"status": "already_running", "camera_id": camera_id}
        else:
            del active_cameras[camera_id]

    redis_client.delete(f"stop_signal_{camera_id}")
    
    # LOGIC ĐIỀU PHỐI: Chọn Queue dựa trên mức độ rủi ro
    # Camera 'high' sẽ vào high_risk_queue để được High_Server xử lý ngay
    target_queue = high_risk_queue if risk_level.lower() == "high" else low_med_queue
    
    print(f"[SYSTEM] Starting {camera_id} on {risk_level.upper()} queue")
    
    p = Process(target=run_camera_streamer, args=(camera_id, rtsp_url, target_queue, risk_level))
    p.start()
    
    active_cameras[camera_id] = p
    return {"status": "started", "camera_id": camera_id, "priority": risk_level}

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
    """
    Truy vấn kết quả mới nhất từ Redis.
    Dữ liệu được cập nhật liên tục bởi các Inference Servers.
    """
    data = redis_client.get(f"cam_status_{camera_id}")
    if not data:
        return {"status": "offline", "message": "No data found in Redis for this camera"}
    return json.loads(data)

@app.get("/system/active_cameras")
def list_cameras():
    running = [cid for cid, p in active_cameras.items() if p.is_alive()]
    return {
        "active_cameras": running,
        "queues": {
            "high_priority": high_risk_queue.qsize(),
            "low_med_priority": low_med_queue.qsize()
        },
        "mode": "dual_inference_pipeline"
    }