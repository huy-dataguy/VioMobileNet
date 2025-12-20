import time
import redis
import json
import cv2
import os
import base64
import numpy as np
import threading
import io
from collections import deque
from minio import Minio

# Kết nối Redis dùng chung
r = redis.Redis(host='redis_server_ai', port=6379, db=0)

# --- Cấu hình MinIO ---
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
# PHẦN 1: INFERENCE SERVER (XỬ LÝ AI TẬP TRUNG)
# =================================================================
def run_inference_server(input_queue, server_label="SHARED_SERVER"):
    print(f"--- [{server_label}] INITIALIZING MODEL ---", flush=True)
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

    print(f"--- [{server_label}] READY (LIFO MODE) ---", flush=True)

    while True:
        q_size = input_queue.qsize()
        if q_size > 15:
            while q_size > 4: 
                try:
                    input_queue.get_nowait()
                    q_size -= 1
                except: break
        
        item = input_queue.get()
        if item is None: continue

        camera_id = item['camera_id']
        frame = item['frame'] 
        timestamp = item['timestamp']

        if camera_id not in cam_states:
            print(f"[{server_label}] New camera detected: {camera_id}", flush=True)
            cam_states[camera_id] = get_clean_state()
            cam_score_buffers[camera_id] = deque(maxlen=3) 
            cam_last_upload[camera_id] = 0

        try:
            t0 = time.time()
            inp = preprocess_frame(frame)
            logits, cam_states[camera_id] = run_inference_step(model, {'image': inp}, cam_states[camera_id])
            
            probs = tf.nn.softmax(logits)
            raw_prob = float(probs[0][0])
            
            # --- LOGIC PHẢN ỨNG CỰC NHANH (BOOST & DROP) ---
            prev_avg = sum(cam_score_buffers[camera_id]) / len(cam_score_buffers[camera_id]) if cam_score_buffers[camera_id] else 0
            cam_score_buffers[camera_id].append(raw_prob)
            
            # Lấy giá trị trung bình trước đó từ bộ đệm
            prev_avg = sum(cam_score_buffers[camera_id]) / len(cam_score_buffers[camera_id]) if cam_score_buffers[camera_id] else 0
            cam_score_buffers[camera_id].append(raw_prob)

            # 1. Tăng vọt khi có dấu hiệu bạo lực (vượt ngưỡng 0.2)
            if raw_prob > 0.2:
                processed_prob = raw_prob ** 0.8
                # Sử dụng Max Pooling để giữ đỉnh cao nhất trong bộ đệm, tránh flicker
                final_prob = max(processed_prob, sum(cam_score_buffers[camera_id])/len(cam_score_buffers[camera_id]))

            # 2. Giảm khi chuyển sang cảnh bình yên 
            elif raw_prob < prev_avg:
                processed_prob = raw_prob ** 3
                final_prob = processed_prob
                
            else:
                processed_prob = raw_prob
                final_prob = processed_prob
            
            is_violent = final_prob > 0.4 
            latency_ms = (time.time() - t0) * 1000
            
            if is_violent:
                r.setex(f"is_violent_status_{camera_id}", 2, "1")
                print(f"[{server_label}][ALERT] {camera_id} ACTIVE | Score: {final_prob:.2f}", flush=True)

            # Threaded Upload
            if is_violent:
                if time.time() - cam_last_upload[camera_id] > 3.0:
                    threading.Thread(
                        target=upload_background, 
                        args=(minio_client, frame.copy(), camera_id, timestamp, server_label)
                    ).start()
                    cam_last_upload[camera_id] = time.time()

            # Cập nhật Redis với TTL dài để phục vụ API
            redis_ttl = 60 if is_violent else 30 
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 25])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            result = {
                "camera_id": camera_id, 
                "is_violent": is_violent, 
                "score": round(final_prob, 4),
                "latency_ms": round(latency_ms, 1),
                "image_preview": jpg_as_text, 
                "status": "violent" if is_violent else "normal",
                "server": server_label,
                "timestamp": timestamp
            }
            r.setex(f"cam_status_{camera_id}", redis_ttl, json.dumps(result))

            processed_count += 1
            if processed_count % 10 == 0:
                delay = (time.time() - timestamp) * 1000
                print(f"[{server_label}] {camera_id} | Infer: {latency_ms:.1f}ms | Delay: {delay:.1f}ms | Q: {input_queue.qsize()}", flush=True)

        except Exception as e:
            print(f"[{server_label} ERROR] {camera_id}: {e}", flush=True)
            cam_states[camera_id] = get_clean_state()

def upload_background(client, frame, c_id, ts, server_label):
    try:
        url = upload_frame_to_minio(client, frame, c_id, ts)
        if url:
            print(f"--- [{server_label} UPLOAD SUCCESS] Evidence for {c_id} saved ---", flush=True)
    except Exception as e:
        print(f"[{server_label} UPLOAD FAILED] {e}", flush=True)

# =================================================================
# PHẦN 2: CAMERA STREAMER (LOGIC ĐẨY FRAME XEN KẼ)
# =================================================================
def run_camera_streamer(camera_id, rtsp_url, input_queue, risk_level):
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    FPS_MAP = {"high": 10, "medium": 4, "low": 2}
    BASE_FPS = FPS_MAP.get(risk_level.lower(), 2)
    
    try:
        cam_num = int(''.join(filter(str.isdigit, camera_id)))
        time.sleep((cam_num % 8) * 0.05)
    except:
        time.sleep(0.1)

    prev_time = 0
    while True:
        if r.get(f"stop_signal_{camera_id}"): break

        is_active_violent = r.get(f"is_violent_status_{camera_id}")
        target_fps = 12 if is_active_violent else BASE_FPS
        frame_interval = 1.0 / target_fps

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1); continue

        now = time.time()
        if now - prev_time >= frame_interval:
            prev_time = now
            small_frame = cv2.resize(frame, (256, 256))
            if not input_queue.full():
                input_queue.put({'camera_id': camera_id, 'frame': small_frame, 'timestamp': now})
                time.sleep(0.005) 
    cap.release()

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