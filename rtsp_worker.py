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
from minio import Minio  # <--- THƯ VIỆN MỚI
from minio.error import S3Error
from core import build_model_optimized, preprocess_frame, run_inference_step

r = redis.Redis(host='redis', port=6379, db=0)
# --- CẤU HÌNH MINIO (Chạy trong Docker VPS) ---

# 1. Lấy thông tin từ biến môi trường (đã khai báo trong docker-compose.yml)
# Nếu không tìm thấy biến môi trường thì dùng giá trị mặc định bên phải
MINIO_INTERNAL_HOST = os.getenv("S3_ENDPOINT_URL", "http://minio:9000").replace("http://", "")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minio")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "mypassword")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "inference-results")

# 2. Cấu hình IP Public để tạo link cho người dùng xem
# Thay bằng IP VPS của bạn
VPS_PUBLIC_IP = "103.78.3.29" 
VPS_MINIO_PORT = "9000"

# Khởi tạo MinIO Client (Kết nối nội bộ Docker)
try:
    minio_client = Minio(
        MINIO_INTERNAL_HOST, # Dùng "minio:9000"
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=False  # <--- QUAN TRỌNG: Đổi thành False vì nội bộ dùng HTTP
    )
    
    # Kiểm tra bucket (thường Docker Compose đã tạo rồi, nhưng check cho chắc)
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)
        # Set policy public để xem được ảnh (nếu chưa set)
        policy = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":["*"]},"Action":["s3:GetObject"],"Resource":["arn:aws:s3:::%s/*"]}]}' % BUCKET_NAME
        minio_client.set_bucket_policy(BUCKET_NAME, policy)
        
    print(f"✅ Đã kết nối tới MinIO Local ({MINIO_INTERNAL_HOST})")
except Exception as e:
    print(f"❌ Lỗi kết nối MinIO: {e}")
    minio_client = None

# --- HÀM UPLOAD MINIO ---
def upload_to_minio_async(frame, camera_id, timestamp):
    def _worker():
        if minio_client is None: return
        
        try:
            # 1. Nén ảnh
            _, buffer = cv2.imencode('.jpg', frame)
            data_stream = io.BytesIO(buffer)
            data_length = len(buffer)
            
            # 2. Đặt tên file
            time_struct = time.localtime(timestamp)
            date_folder = time.strftime("%Y-%m-%d", time_struct)
            filename = f"{camera_id}/{date_folder}/{int(timestamp*1000)}.jpg"
            
            # 3. Upload (Dùng mạng nội bộ Docker - siêu nhanh)
            minio_client.put_object(
                BUCKET_NAME,
                filename,
                data_stream,
                data_length,
                content_type="image/jpeg"
            )
            
        except Exception as e:
            print(f"MinIO Upload Error: {e}")

    threading.Thread(target=_worker).start()
    
    # 4. Trả về Link Public (Dùng IP VPS 103.78... để Client xem được)
    # URL này dành cho trình duyệt/App bên ngoài
    return f"http://{VPS_PUBLIC_IP}:{VPS_MINIO_PORT}/{BUCKET_NAME}/{filename}"
    
# --- CLASS RTSPStream (Giữ nguyên) ---
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

# --- MAIN WORKER ---
def run_camera_process(camera_id, rtsp_url):
    print(f"--- [WORKER {camera_id}] MINIO ENABLED ---")
    
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try: tf.config.experimental.set_memory_growth(gpu, True)
        except: pass

    model = build_model_optimized()
    
    def get_clean_state():
        return model.init_states(tf.shape(tf.ones([1, 1, 256, 256, 3])))

    states = get_clean_state()
    stream = RTSPStream(rtsp_url)
    
    TARGET_FPS = 12 
    frame_duration = 1.0 / TARGET_FPS
    prev_inference_time = 0
    score_buffer = deque(maxlen=3) 
    
    prev_frame_gray = None
    frame_count = 0
    RESET_INTERVAL = 3000
    
    # Metrics
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0
    inference_time_ms = 0.0
    alert_active = False # Trạng thái đang báo động
    
    time.sleep(1) 
    
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
        
        # FPS calc
        fps_counter += 1
        if now - fps_start_time >= 1.0:
            current_fps = fps_counter / (now - fps_start_time)
            fps_counter = 0
            fps_start_time = now
        
        # Scene Change
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
            prev_frame_gray = gray
        except: pass

        frame_count += 1
        if frame_count >= RESET_INTERVAL and not reset_trigger:
            states = get_clean_state()
            score_buffer.clear()
            frame_count = 0
            gc.collect()
            
        try:
            t0 = time.time()
            inp = preprocess_frame(frame)
            inputs = {'image': inp}
            logits, states = run_inference_step(model, inputs, states)
            probs = tf.nn.softmax(logits)
            raw_prob = float(probs[0][0])
            inference_time_ms = (time.time() - t0) * 1000 
            
            if raw_prob > 0.85:
                 final_prob = raw_prob
            else:
                if not reset_trigger: score_buffer.append(raw_prob)
                final_prob = sum(score_buffer) / len(score_buffer) if score_buffer else raw_prob

            is_violent = final_prob > 0.7
            
            # --- LOGIC GỬI ẢNH VỀ MINIO TRÊN LAPTOP ---
            evidence_url = None
            
            # Chỉ gửi ảnh khi có bạo lực
            if is_violent:
                # Chiến lược: Mỗi giây gửi 1 ảnh (để tránh spam đường truyền)
                if int(now * 10) % 10 == 0: 
                    # Gửi frame gốc (Full HD) về laptop
                    # Dùng copy() để không bị xung đột luồng
                    evidence_url = upload_to_minio_async(frame.copy(), camera_id, now)
            # ------------------------------------------

            # Gửi Client xem trước (Ảnh nhỏ Base64)
            frame_view = cv2.resize(frame, (320, 180)) 
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
            _, buffer = cv2.imencode('.jpg', frame_view, encode_param)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            result = {
                "camera_id": camera_id,
                "is_violent": is_violent,
                "score": round(final_prob, 4),
                "fps": round(current_fps, 1),
                "latency_ms": round(inference_time_ms, 1),
                "image_preview": jpg_as_text, # Xem nhanh
                "evidence_url": evidence_url, # Link tải ảnh gốc từ MinIO
                "timestamp": now
            }
            
            r.setex(f"cam_status_{camera_id}", 5, json.dumps(result))
            
            if final_prob > 0.6:
                print(f"[{camera_id}] Violence: {final_prob:.2f}")

        except Exception as e:
            print(f"Error: {e}")
            states = get_clean_state()

    stream.stop()
    r.delete(f"cam_status_{camera_id}")
    print(f"[{camera_id}] Stopped.")
