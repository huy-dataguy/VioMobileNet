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


def run_camera_process(camera_id, rtsp_url):
    """
    Hàm này chạy trong một Process riêng biệt.
    Tại đây mới bắt đầu import TensorFlow để cô lập bộ nhớ.
    """
    print(f"--- [WORKER {camera_id}] INITIALIZING ---")
    
    # --- LAZY IMPORT (Chỉ load khi process chạy) ---
    import tensorflow as tf
    from core import build_model_optimized, preprocess_frame, run_inference_step, setup_gpu_config
    
    # 1. Setup GPU ngay lập tức cho Process này
    setup_gpu_config()

    # Kết nối Redis (cần tạo kết nối mới trong process con)
    r = redis.Redis(host='redis', port=6379, db=0)

    # --- Cấu hình MinIO (Copy lại logic cũ) ---
    MINIO_INTERNAL_HOST = os.getenv("S3_ENDPOINT_URL", "minio-vps:9000").replace("http://", "")
    ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minio")
    SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "mypassword")
    BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "inference-results")
    VPS_PUBLIC_IP = "192.168.1.5" 
    VPS_MINIO_PORT = "9000"

    minio_client = None
    try:
        minio_client = Minio(MINIO_INTERNAL_HOST, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)
        if not minio_client.bucket_exists(BUCKET_NAME):
            minio_client.make_bucket(BUCKET_NAME)
            policy = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":["*"]},"Action":["s3:GetObject"],"Resource":["arn:aws:s3:::%s/*"]}]}' % BUCKET_NAME
            minio_client.set_bucket_policy(BUCKET_NAME, policy)
    except Exception as e:
        print(f"[{camera_id}] MinIO Error: {e}")

    def upload_to_minio_async(frame, c_id, ts):
        time_struct = time.localtime(ts)
        date_folder = time.strftime("%Y-%m-%d", time_struct)
        filename = f"{c_id}/{date_folder}/{int(ts*1000)}.jpg"
        def _worker():
            if minio_client is None: return
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                minio_client.put_object(BUCKET_NAME, filename, io.BytesIO(buffer), len(buffer), content_type="image/jpeg")
            except: pass
        threading.Thread(target=_worker).start()
        return f"http://{VPS_PUBLIC_IP}:{VPS_MINIO_PORT}/{BUCKET_NAME}/{filename}"

    class RTSPStream:
        def __init__(self, src):
            self.capture = cv2.VideoCapture(src)
            self.frame = None
            self.status, self.frame = self.capture.read()
            self.stopped = False
            self.t = threading.Thread(target=self.update, daemon=True)
            self.t.start()
        def update(self):
            while not self.stopped:
                status, frame = self.capture.read()
                if status: self.frame = frame; self.status = True
                else: self.status = False
                time.sleep(0.001)
        def read(self): return self.status, self.frame
        def stop(self): self.stopped = True; self.t.join(); self.capture.release()

    try:
        model = build_model_optimized()
        
        def get_clean_state():
            return model.init_states(tf.shape(tf.ones([1, 1, 256, 256, 3])))

        states = get_clean_state()
        stream = RTSPStream(rtsp_url)
        
        prev_inference_time = 0
        frame_duration = 1.0 / 12 # 12 FPS
        score_buffer = deque(maxlen=3)
        prev_frame_gray = None
        frame_count = 0
        
        fps_counter, fps_start_time = 0, time.time()
        current_fps, inference_time_ms = 0.0, 0.0
        cached_evidence_url, last_upload_time = None, 0

        time.sleep(1) # Warmup

        print(f"[{camera_id}] Started Processing Loop")
        
        while True:
            # Check Redis signal stop
            if r.get(f"stop_signal_{camera_id}"): 
                print(f"[{camera_id}] Stop signal received.")
                break
                
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(1)
                continue
                
            now = time.time()
            if now - prev_inference_time < frame_duration:
                time.sleep(0.005); continue
            prev_inference_time = now
            
            # FPS & Scene Logic
            fps_counter += 1
            if now - fps_start_time >= 1.0:
                current_fps = fps_counter / (now - fps_start_time)
                fps_counter = 0; fps_start_time = now

            # Scene Change Reset
            try:
                small = cv2.resize(frame, (64, 64))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev_frame_gray is not None:
                    if np.mean(cv2.absdiff(gray, prev_frame_gray)) > 35.0:
                        states = get_clean_state(); score_buffer.clear()
                        cached_evidence_url = None
                prev_frame_gray = gray
            except: pass
            
            # Memory Auto Reset
            frame_count += 1
            if frame_count >= 3000:
                states = get_clean_state(); score_buffer.clear(); frame_count = 0; gc.collect()

            # INFERENCE
            try:
                t0 = time.time()
                inp = preprocess_frame(frame)
                inputs = {'image': inp}
                logits, states = run_inference_step(model, inputs, states)
                probs = tf.nn.softmax(logits)
                raw_prob = float(probs[0][0])
                inference_time_ms = (time.time() - t0) * 1000
                
                # Logic Buffer
                if raw_prob > 0.85: final_prob = raw_prob
                else:
                    score_buffer.append(raw_prob)
                    final_prob = sum(score_buffer) / len(score_buffer) if score_buffer else raw_prob
                
                is_violent = final_prob > 0.7
                
                # Upload Logic
                current_evidence_url = None
                if is_violent:
                    if cached_evidence_url is None or (now - last_upload_time >= 1.0):
                        cached_evidence_url = upload_to_minio_async(frame.copy(), camera_id, now)
                        last_upload_time = now
                    current_evidence_url = cached_evidence_url
                else:
                    cached_evidence_url = None

                # Preview
                frame_view = cv2.resize(frame, (320, 180))
                _, buffer = cv2.imencode('.jpg', frame_view, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                result = {
                    "camera_id": camera_id, "is_violent": is_violent, "score": round(final_prob, 4),
                    "fps": round(current_fps, 1), "latency_ms": round(inference_time_ms, 1),
                    "image_preview": jpg_as_text, "evidence_url": current_evidence_url, "timestamp": now
                }
                r.setex(f"cam_status_{camera_id}", 5, json.dumps(result))
                
            except Exception as e:
                print(f"[{camera_id}] Inf Error: {e}")
                states = get_clean_state()

        stream.stop()
        r.delete(f"cam_status_{camera_id}")
    except Exception as e:
        print(f"[{camera_id}] CRITICAL FAIL: {e}")
    finally:
        print(f"[{camera_id}] Cleanup & Exit")
