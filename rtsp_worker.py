import time
import redis
import json
import cv2
import os
import gc
import base64
import numpy as np
import threading
import io
from collections import deque
from minio import Minio

# Kết nối Redis dùng chung
r = redis.Redis(host='redis_server_ai', port=6379, db=0)

# --- Cấu hình MinIO dùng chung ---
MINIO_INTERNAL_HOST = os.getenv("S3_ENDPOINT_URL", "minio:9000").replace("http://", "")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minio")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "mypassword")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "inference-results")
VPS_PUBLIC_IP = "192.168.0.200"
VPS_MINIO_PORT = "9000"

def get_minio_client():
    try:
        return Minio(MINIO_INTERNAL_HOST, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)
    except:
        return None

# =================================================================
# PHẦN 1: INFERENCE SERVER (CHẠY DUY NHẤT 1 BẢN TRONG RAM)
# =================================================================
def run_inference_server(input_queue):
    """
    Inference Server tối ưu theo cơ chế LIFO.
    Cập nhật: 
    - Log ALERT liên tục khi có bạo lực.
    - Ép in log (flush) để tránh delay log.
    - Điều chỉnh TTL Redis: 1s khi bạo lực, 10s khi bình thường.
    """
    print("--- [SERVER] INITIALIZING SHARED MODEL (LIFO MODE) ---", flush=True)
    import tensorflow as tf
    from core import build_model_optimized, preprocess_frame, run_inference_step, setup_gpu_config
    
    setup_gpu_config()
    model = build_model_optimized()
    minio_client = get_minio_client()

    def get_clean_state():
        return model.init_states(tf.shape(tf.ones([1, 1, 256, 256, 3])))

    cam_states = {}
    cam_score_buffers = {}
    cam_last_upload = {}
    processed_count = 0

    print("--- [SERVER] LIFO SHARED MODEL READY ---", flush=True)

    while True:
        # --- CHIẾN THUẬT LIFO: BỎ QUA FRAME CŨ ---
        q_size = input_queue.qsize()
        item = None
        
        if q_size > 0:
            for _ in range(q_size):
                item = input_queue.get()
        else:
            item = input_queue.get()

        if item is None: continue

        camera_id = item['camera_id']
        frame = item['frame'] 
        timestamp = item['timestamp']

        if camera_id not in cam_states:
            print(f"[DEBUG LIFO] New camera detected: {camera_id}", flush=True)
            cam_states[camera_id] = get_clean_state()
            cam_score_buffers[camera_id] = deque(maxlen=3)
            cam_last_upload[camera_id] = 0

        try:
            t0 = time.time()
            inp = preprocess_frame(frame)
            logits, cam_states[camera_id] = run_inference_step(model, {'image': inp}, cam_states[camera_id])
            
            probs = tf.nn.softmax(logits)
            raw_prob = float(probs[0][0])
            
            # Max Pooling Score để tăng độ nhạy
            cam_score_buffers[camera_id].append(raw_prob)
            final_prob = max(cam_score_buffers[camera_id])
            
            is_violent = final_prob > 0.5
            latency_ms = (time.time() - t0) * 1000
            
            # 1. LOG ALERT BẠO LỰC LIÊN TỤC (Tách khỏi logic upload)
            if is_violent:
                print(f"[ALERT] {camera_id} VIOLENCE ACTIVE | Score: {final_prob:.2f} | Time: {time.strftime('%H:%M:%S')}", flush=True)

            # 2. LOG DEBUG ĐỊNH KỲ (Mỗi 5 frame để mượt hơn)
            processed_count += 1
            if processed_count % 5 == 0:
                pipeline_delay = (time.time() - timestamp) * 1000
                print(f"[LIFO] {camera_id} | Infer: {latency_ms:.1f}ms | Delay: {pipeline_delay:.1f}ms | Skipped: {q_size-1}", flush=True)

            # 3. LOG UPLOAD ẢNH (Cooldown 3s để tiết kiệm tài nguyên MinIO)
            evidence_url = None
            if is_violent:
                if time.time() - cam_last_upload[camera_id] > 3.0:
                    print(f"--- [UPLOAD] Saving evidence for {camera_id} ---", flush=True)
                    evidence_url = upload_frame_to_minio(minio_client, frame, camera_id, timestamp)
                    cam_last_upload[camera_id] = time.time()

            # 4. CẬP NHẬT KẾT QUẢ LÊN REDIS (Điều chỉnh TTL theo yêu cầu)
            # Nếu bạo lực: TTL 1s (cập nhật nhanh)
            # Nếu bình thường: TTL 10s
            redis_ttl = 1 if is_violent else 10

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            result = {
                "camera_id": camera_id, 
                "is_violent": is_violent, 
                "score": round(final_prob, 4),
                "latency_ms": round(latency_ms, 1),
                "image_preview": jpg_as_text, 
                "evidence_url": evidence_url, 
                "timestamp": timestamp,
                "status": "violent" if is_violent else "normal"
            }
            
            # Set kết quả với TTL tương ứng
            r.setex(f"cam_status_{camera_id}", redis_ttl, json.dumps(result))

        except Exception as e:
            print(f"[LIFO ERROR] {camera_id}: {e}", flush=True)
            cam_states[camera_id] = get_clean_state()

# =================================================================
# PHẦN 2: CAMERA STREAMER (CHẠY NHIỀU BẢN, CỰC NHẸ)
# =================================================================
def run_camera_streamer(camera_id, rtsp_url, input_queue, risk_level):
    print(f"--- [STREAMER {camera_id}] STARTING WITH RISK: {risk_level} ---")
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ĐỊNH NGHĨA FPS THEO RISK LEVEL
    if risk_level.lower() == "high":
        TARGET_FPS = 12  # Khu vực nóng: Xử lý dày để không sót
    elif risk_level.lower() == "medium":
        TARGET_FPS = 5   # Khu vực trung bình
    else:
        TARGET_FPS = 2   # Khu vực an toàn: Xử lý thưa để cứu CPU

    frame_interval = 1.0 / TARGET_FPS
    prev_time = 0

    while True:
        if r.get(f"stop_signal_{camera_id}"): 
            break

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1); continue

        now = time.time()
        if now - prev_time >= frame_interval:
            prev_time = now
            small_frame = cv2.resize(frame, (256, 256))
            
            if not input_queue.full():
                input_queue.put({
                    'camera_id': camera_id, 
                    'frame': small_frame,
                    'timestamp': now
                })

    cap.release()
    print(f"--- [STREAMER {camera_id}] STOPPED ---")

# =================================================================
# HÀM BỔ TRỢ
# =================================================================
def upload_frame_to_minio(client, frame, c_id, ts):
    if client is None: return None
    try:
        time_struct = time.localtime(ts)
        date_folder = time.strftime("%Y-%m-%d", time_struct)
        filename = f"{c_id}/{date_folder}/{int(ts*1000)}.jpg"
        
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        client.put_object(BUCKET_NAME, filename, io.BytesIO(buffer), len(buffer), content_type="image/jpeg")
        return f"http://{VPS_PUBLIC_IP}:{VPS_MINIO_PORT}/{BUCKET_NAME}/{filename}"
    except:
        return None