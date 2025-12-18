from fastapi import FastAPI, UploadFile, File
import shutil
import os
import redis
import json
from celery import Celery
from multiprocessing import Process


from rtsp_worker import run_camera_process

app = FastAPI(title="MoViNet Video & RTSP System")

redis_client = redis.Redis(host='redis_server_ai', port=6379, db=0)
celery_client = Celery('video_tasks', broker='redis://redis_server_ai:6379/0', backend='redis://redis_server_ai:6379/0')

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
active_cameras = {} 

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
def start_camera(camera_id: str, rtsp_url: str):
    if camera_id in active_cameras:
        # Ép buộc dừng và xóa tiến trình cũ nếu nó không phản hồi
        active_cameras[camera_id].terminate()
        active_cameras[camera_id].join(timeout=1)
        del active_cameras[camera_id]
        print(f"Cleaning old zombie process for {camera_id}")
    
    redis_client.delete(f"stop_signal_{camera_id}")
    p = Process(target=run_camera_process, args=(camera_id, rtsp_url))
    p.start()
    active_cameras[camera_id] = p
    return {"status": "started", "pid": p.pid}

@app.post("/camera/stop")
def stop_camera(camera_id: str):
    if camera_id not in active_cameras:
        return {"status": "error", "message": "Camera not found"}
    
    redis_client.set(f"stop_signal_{camera_id}", "1")
    
    active_cameras[camera_id].join(timeout=2)
    
    if active_cameras[camera_id].is_alive():
        print(f"Force killing camera {camera_id}")
        active_cameras[camera_id].terminate() 
        
    del active_cameras[camera_id]
    return {"status": "stopped"}

@app.get("/camera/status/{camera_id}")
def get_cam_status(camera_id: str):
    data = redis_client.get(f"cam_status_{camera_id}")
    if not data:
        return {"status": "offline", "message": "No signal"}
    return json.loads(data)

@app.get("/system/active_cameras")
def list_cameras():
    running = [cid for cid, p in active_cameras.items() if p.is_alive()]
    return {"active_cameras": running}