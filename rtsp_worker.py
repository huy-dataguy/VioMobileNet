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
    Hàm xử lý worker cho từng camera đã được tối ưu hóa CPU và RAM.
    """
    print(f"--- [WORKER {camera_id}] INITIALIZING OPTIMIZED PROCESS ---")
    
    # --- LAZY IMPORT ---
    import tensorflow as tf
    from core import build_model_optimized, preprocess_frame, run_inference_step, setup_gpu_config
    
    # 1. Giới hạn tài nguyên CPU cho mỗi Process 
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    setup_gpu_config()

    r = redis.Redis(host='redis_server_ai', port=6379, db=0)

    # --- Cấu hình MinIO ---
    MINIO_INTERNAL_HOST = os.getenv("S3_ENDPOINT_URL", "minio:9000").replace("http://", "")
    ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minio")
    SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "mypassword")
    BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "inference-results")
    VPS_PUBLIC_IP = "192.168.0.200" 
    VPS_MINIO_PORT = "9000"

    minio_client = None
    try:
        minio_client = Minio(MINIO_INTERNAL_HOST, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)
    except Exception as e:
        print(f"[{camera_id}] MinIO Init Error: {e}")

    def upload_to_minio_async(frame, c_id, ts):
        time_struct = time.localtime(ts)
        date_folder = time.strftime("%Y-%m-%d", time_struct)
        filename = f"{c_id}/{date_folder}/{int(ts*1000)}.jpg"
        def _worker():
            if minio_client is None: return
            try:
                # Resize ảnh bằng chứng xuống 480p để tiết kiệm băng thông và disk I/O
                small_evidence = cv2.resize(frame, (640, 480))
                _, buffer = cv2.imencode('.jpg', small_evidence, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                minio_client.put_object(BUCKET_NAME, filename, io.BytesIO(buffer), len(buffer), content_type="image/jpeg")
            except: pass
        threading.Thread(target=_worker, daemon=True).start()
        return f"http://{VPS_PUBLIC_IP}:{VPS_MINIO_PORT}/{BUCKET_NAME}/{filename}"

    class RTSPStream:
        def __init__(self, src):
            # Tối ưu FFmpeg: Giảm buffer size và tăng tốc độ giải mã
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|sdp_backlog;20'
            self.capture = cv2.VideoCapture(src)
            self.frame = None
            self.status = False
            self.stopped = False
            self.t = threading.Thread(target=self.update, daemon=True)
            self.t.start()

        def update(self):
            while not self.stopped:
                # Đọc frame liên tục để clear buffer của camera
                status, frame = self.capture.read()
                if status:
                    self.frame = frame
                    self.status = True
                else:
                    self.status = False
                    time.sleep(0.1)

        def read(self): 
            return self.status, self.frame

        def stop(self): 
            self.stopped = True
            self.capture.release()

    try:
        model = build_model_optimized()
        def get_clean_state():
            return model.init_states(tf.shape(tf.ones([1, 1, 256, 256, 3])))

        states = get_clean_state()
        stream = RTSPStream(rtsp_url)
        
        # --- THAY ĐỔI QUAN TRỌNG: Giảm tần suất Inference ---
        # Chỉ chạy 5 FPS (thay vì 12 hoặc 30). Bạo lực không cần phân tích quá dày.
        TARGET_INFERENCE_FPS = 5 
        frame_duration = 1.0 / TARGET_INFERENCE_FPS
        
        prev_inference_time = 0
        score_buffer = deque(maxlen=3)
        prev_frame_gray = None
        frame_count = 0
        fps_counter, fps_start_time = 0, time.time()
        current_fps, inference_time_ms = 0.0, 0.0
        cached_evidence_url, last_upload_time = None, 0

        print(f"[{camera_id}] Started Optimized Loop (Target: {TARGET_INFERENCE_FPS} FPS)")
        
        while True:
            if r.get(f"stop_signal_{camera_id}"): break
                
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.1); continue
                
            now = time.time()
            # Kiểm tra xem đã đến lúc cần phân tích chưa (Skip frames logic)
            if now - prev_inference_time < frame_duration:
                continue 
            
            prev_inference_time = now
            fps_counter += 1

            # 1. Resize ảnh đầu vào ngay lập tức để các bước sau (Scene change, Preview) nhẹ hơn
            # frame_small dùng cho các logic phụ
            frame_small = cv2.resize(frame, (320, 240))

            # Scene Change Reset (Dùng frame_small để tiết kiệm CPU)
            if fps_counter % 5 == 0: # Chỉ check scene change mỗi giây 1 lần
                try:
                    gray = cv2.cvtColor(cv2.resize(frame_small, (64, 64)), cv2.COLOR_BGR2GRAY)
                    if prev_frame_gray is not None:
                        if np.mean(cv2.absdiff(gray, prev_frame_gray)) > 40.0:
                            states = get_clean_state(); score_buffer.clear()
                    prev_frame_gray = gray
                except: pass

            # Update FPS Monitor
            if now - fps_start_time >= 1.0:
                current_fps = fps_counter / (now - fps_start_time)
                fps_counter = 0; fps_start_time = now

            # INFERENCE
            try:
                t0 = time.time()
                # Hàm preprocess_frame nên nhận frame đã resize nếu được
                inp = preprocess_frame(frame) 
                logits, states = run_inference_step(model, {'image': inp}, states)
                
                probs = tf.nn.softmax(logits)
                raw_prob = float(probs[0][0])
                inference_time_ms = (time.time() - t0) * 1000
                
                # Logic Buffer ổn định kết quả
                score_buffer.append(raw_prob)
                final_prob = sum(score_buffer) / len(score_buffer)
                is_violent = final_prob > 0.75 # Tăng ngưỡng để giảm False Positive
                
                # Upload Logic
                current_evidence_url = None
                if is_violent:
                    if cached_evidence_url is None or (now - last_upload_time >= 2.0):
                        cached_evidence_url = upload_to_minio_async(frame, camera_id, now)
                        last_upload_time = now
                    current_evidence_url = cached_evidence_url
                else:
                    cached_evidence_url = None

                # Preview: Dùng frame_small đã có sẵn, giảm chất lượng JPEG để nhẹ Redis
                _, buffer = cv2.imencode('.jpg', frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                result = {
                    "camera_id": camera_id, "is_violent": is_violent, "score": round(final_prob, 4),
                    "fps": round(current_fps, 1), "latency_ms": round(inference_time_ms, 1),
                    "image_preview": jpg_as_text, "evidence_url": current_evidence_url, "timestamp": now
                }
                r.setex(f"cam_status_{camera_id}", 5, json.dumps(result))
                
            except Exception as e:
                states = get_clean_state()

            # GC & Memory cleanup mỗi 1000 frames
            frame_count += 1
            if frame_count % 1000 == 0:
                gc.collect()

    except Exception as e:
        print(f"[{camera_id}] CRITICAL FAIL: {e}")
    finally:
        stream.stop()
        r.delete(f"cam_status_{camera_id}")
        print(f"[{camera_id}] Cleanup & Exit")