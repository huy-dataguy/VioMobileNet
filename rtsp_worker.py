import time
import redis
import json
import cv2
import os
import tensorflow as tf
import gc
import base64
import numpy as np
import threading
import io
from collections import deque
from minio import Minio 
from minio.error import S3Error
from core import build_model_optimized, preprocess_frame, run_inference_step

# Kết nối Redis
r = redis.Redis(host='redis', port=6379, db=0)

# ================= CẤU HÌNH MINIO =================
# 1. Lấy thông tin từ biến môi trường (trong docker-compose)
MINIO_INTERNAL_HOST = os.getenv("S3_ENDPOINT_URL", "minio:9000").replace("http://", "")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minio")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "mypassword")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "inference-results")

# 2. Cấu hình IP Public để tạo link cho người dùng xem
# Thay bằng IP VPS thật của bạn (hoặc domain ngrok nếu dùng local)
VPS_PUBLIC_IP = "103.78.3.29" 
VPS_MINIO_PORT = "9000"

# Khởi tạo MinIO Client
try:
    minio_client = Minio(
        MINIO_INTERNAL_HOST,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=False 
    )
    
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)
        # Set policy public
        policy = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":["*"]},"Action":["s3:GetObject"],"Resource":["arn:aws:s3:::%s/*"]}]}' % BUCKET_NAME
        minio_client.set_bucket_policy(BUCKET_NAME, policy)
        
    print(f"✅ [MinIO] Connected to {MINIO_INTERNAL_HOST}")
except Exception as e:
    print(f"❌ [MinIO] Connection Failed: {e}")
    minio_client = None

# ================= HÀM UPLOAD ASYNC (ĐÃ FIX) =================
def upload_to_minio_async(frame, camera_id, timestamp):
    # 1. TẠO TÊN FILE Ở NGOÀI (Fix lỗi NameError)
    time_struct = time.localtime(timestamp)
    date_folder = time.strftime("%Y-%m-%d", time_struct)
    filename = f"{camera_id}/{date_folder}/{int(timestamp*1000)}.jpg"
    
    def _worker():
        if minio_client is None: return
        try:
            # Nén ảnh
            _, buffer = cv2.imencode('.jpg', frame)
            data_stream = io.BytesIO(buffer)
            data_length = len(buffer)
            
            # Upload
            minio_client.put_object(
                BUCKET_NAME,
                filename,
                data_stream,
                data_length,
                content_type="image/jpeg"
            )
        except Exception as e:
            print(f"MinIO Upload Error: {e}")

    # Chạy thread ngầm
    threading.Thread(target=_worker).start()
    
    # Trả về URL ngay lập tức
    return f"http://{VPS_PUBLIC_IP}:{VPS_MINIO_PORT}/{BUCKET_NAME}/{filename}"

# ================= CLASS RTSP STREAM =================
class RTSPStream:
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.frame = None
        self.status = False
        self.stopped = False
        self.status, self.frame = self.capture.read()
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()

    def update(self):
        while True:
            if self.stopped: return
            status, frame = self.capture.read()
            if status:
                self.frame = frame
                self.status = True
            else:
                self.status = False
            time.sleep(0.001)

    def read(self):
        return self.status, self.frame

    def stop(self):
        self.stopped = True
        self.t.join()
        self.capture.release()

# ================= MAIN WORKER PROCESS =================
def run_camera_process(camera_id, rtsp_url):
    print(f"--- [WORKER {camera_id}] STARTED ---")
    
    # Config GPU
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try: tf.config.experimental.set_memory_growth(gpu, True)
        except: pass

    model = build_model_optimized()
    
    def get_clean_state():
        return model.init_states(tf.shape(tf.ones([1, 1, 256, 256, 3])))

    states = get_clean_state()
    stream = RTSPStream(rtsp_url)
    
    # Params
    TARGET_FPS = 12 
    frame_duration = 1.0 / TARGET_FPS
    prev_inference_time = 0
    score_buffer = deque(maxlen=3) 
    
    prev_frame_gray = None
    frame_count = 0
    RESET_INTERVAL = 3000
    
    # Metrics Variables
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0
    inference_time_ms = 0.0
    
    # --- BIẾN DÍNH (FIX LỖI NULL URL) ---
    cached_evidence_url = None   # Lưu URL gần nhất
    last_upload_time = 0         # Thời điểm upload cuối cùng
    
    time.sleep(1) # Warmup
    
    while True:
        if r.get(f"stop_signal_{camera_id}"): break
            
        ret, frame = stream.read()
        if not ret or frame is None:
            time.sleep(1)
            continue
            
        now = time.time()
        if now - prev_inference_time < frame_duration:
            time.sleep(0.005)
            continue
            
        prev_inference_time = now
        
        # FPS Calculation
        fps_counter += 1
        if now - fps_start_time >= 1.0:
            current_fps = fps_counter / (now - fps_start_time)
            fps_counter = 0
            fps_start_time = now
        
        # Scene Change Detection
        reset_trigger = False
        try:
            small_frame = cv2.resize(frame, (64, 64))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            if prev_frame_gray is not None:
                score_diff = cv2.absdiff(gray, prev_frame_gray)
                if np.mean(score_diff) > 35.0:
                    states = get_clean_state()
                    score_buffer.clear()
                    reset_trigger = True
                    # Reset cache khi đổi cảnh
                    cached_evidence_url = None 
            prev_frame_gray = gray
        except: pass

        # Auto Reset Memory
        frame_count += 1
        if frame_count >= RESET_INTERVAL and not reset_trigger:
            states = get_clean_state()
            score_buffer.clear()
            frame_count = 0
            gc.collect()
            
        try:
            # --- INFERENCE ---
            t0 = time.time()
            inp = preprocess_frame(frame)
            inputs = {'image': inp}
            logits, states = run_inference_step(model, inputs, states)
            probs = tf.nn.softmax(logits)
            raw_prob = float(probs[0][0])
            inference_time_ms = (time.time() - t0) * 1000 
            
            # Logic tính điểm
            if raw_prob > 0.85:
                 final_prob = raw_prob
            else:
                if not reset_trigger: score_buffer.append(raw_prob)
                final_prob = sum(score_buffer) / len(score_buffer) if score_buffer else raw_prob

            is_violent = final_prob > 0.7
            
            # --- LOGIC EVIDENCE URL (FIXED) ---
            current_evidence_url = None
            
            if is_violent:
                # Nếu chưa có cache HOẶC đã quá 1 giây kể từ lần upload trước
                if cached_evidence_url is None or (now - last_upload_time >= 1.0):
                    # Upload ảnh mới và cập nhật cache
                    cached_evidence_url = upload_to_minio_async(frame.copy(), camera_id, now)
                    last_upload_time = now
                
                # Luôn dùng giá trị trong cache để gửi đi (kể cả khi không upload ở frame này)
                current_evidence_url = cached_evidence_url
            else:
                # Hết đánh nhau -> Xóa cache
                cached_evidence_url = None
                current_evidence_url = None
            # ----------------------------------

            # Preview Image (Base64 Small)
            frame_view = cv2.resize(frame, (320, 180)) 
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
            _, buffer = cv2.imencode('.jpg', frame_view, encode_param)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # Result JSON
            result = {
                "camera_id": camera_id,
                "is_violent": is_violent,
                "score": round(final_prob, 4),
                "fps": round(current_fps, 1),
                "latency_ms": round(inference_time_ms, 1),
                "image_preview": jpg_as_text,
                "evidence_url": current_evidence_url, # Giá trị này giờ sẽ ổn định
                "timestamp": now
            }
            
            r.setex(f"cam_status_{camera_id}", 5, json.dumps(result))
            
            if final_prob > 0.6:
                print(f"[{camera_id}] 🚨 Violence: {final_prob:.2f}")

        except Exception as e:
            print(f"Error: {e}")
            states = get_clean_state()

    stream.stop()
    r.delete(f"cam_status_{camera_id}")
    print(f"[{camera_id}] Stopped.")
