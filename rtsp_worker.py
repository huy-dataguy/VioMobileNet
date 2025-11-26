# import time
# import redis
# import json
# import cv2
# import tensorflow as tf
# import gc
# from core import build_model_optimized, preprocess_frame, run_inference_step

# r = redis.Redis(host='redis', port=6379, db=0)

# def run_camera_process(camera_id, rtsp_url):
#     print(f"--- [RTSP WORKER {camera_id}] STARTING ---")
    
#     # Cấu hình GPU Memory Growth để tránh chiếm hết VRAM
#     gpus = tf.config.list_physical_devices('GPU')
#     for gpu in gpus:
#         try:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         except: pass

#     # Load Model riêng cho process này
#     model = build_model_optimized()
    
#     def get_clean_state():
#         return model.init_states(tf.shape(tf.ones([1, 1, 256, 256, 3])))

#     states = get_clean_state()
#     cap = cv2.VideoCapture(rtsp_url)
    
#     prev_time = 0
#     fps_limit = 10  # Giới hạn FPS xử lý để tiết kiệm GPU
#     frame_count = 0
#     RESET_INTERVAL = 5000 # Reset state sau mỗi 5000 frames (~8 phút)
    
#     while True:
#         # 1. Kiểm tra tín hiệu dừng từ Redis
#         if r.get(f"stop_signal_{camera_id}"):
#             print(f"[{camera_id}] Stop signal received.")
#             break
            
#         ret, frame = cap.read()
#         if not ret:
#             print(f"[{camera_id}] Lost signal, retrying...")
#             time.sleep(2)
#             cap = cv2.VideoCapture(rtsp_url)
#             states = get_clean_state() # Reset state khi mất tín hiệu
#             continue
            
#         # 2. Kiểm soát FPS
#         now = time.time()
#         if now - prev_time < 1./fps_limit:
#             continue
#         prev_time = now
        
#         # 3. Auto Reset State (Tránh lỗi tích lũy sai số)
#         frame_count += 1
#         if frame_count >= RESET_INTERVAL:
#             states = get_clean_state()
#             frame_count = 0
#             gc.collect() # Dọn rác RAM
            
#         # 4. Inference
#         try:
#             inp = preprocess_frame(frame)
#             inputs = {'image': inp}
            
#             # Gọi hàm tf.function từ core
#             logits, states = run_inference_step(model, inputs, states)
            
#             probs = tf.nn.softmax(logits)
#             fight_prob = float(probs[0][0])
            
#             # 5. Ghi kết quả vào Redis (API sẽ đọc từ đây)
#             result = {
#                 "camera_id": camera_id,
#                 "fight_prob": fight_prob,
#                 "is_violent": fight_prob > 0.7,
#                 "timestamp": now
#             }
#             # Key tồn tại trong 5 giây
#             r.setex(f"cam_status_{camera_id}", 5, json.dumps(result))
            
#             if fight_prob > 0.8:
#                 print(f"!!! VIOLENCE DETECTED [{camera_id}]: {fight_prob:.2f}")

#         except Exception as e:
#             print(f"Error inference: {e}")
#             states = get_clean_state()

#     cap.release()
#     r.delete(f"cam_status_{camera_id}")
#     print(f"[{camera_id}] Stopped.")


import time
import redis
import json
import cv2
import tensorflow as tf
import gc
import base64
import numpy as np
from collections import deque
from core import build_model_optimized, preprocess_frame, run_inference_step

r = redis.Redis(host='redis', port=6379, db=0)

def run_camera_process(camera_id, rtsp_url):
    print(f"--- [RTSP WORKER {camera_id}] STARTING OPTIMIZED ---")
    
    # 1. Config GPU
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except: pass

    # 2. Load Model
    model = build_model_optimized()
    
    def get_clean_state():
        return model.init_states(tf.shape(tf.ones([1, 1, 256, 256, 3])))

    states = get_clean_state()
    
    # Mở Camera
    cap = cv2.VideoCapture(rtsp_url)
    
    # --- CẤU HÌNH QUAN TRỌNG ---
    # Model A3 train ở 12 FPS -> Phải để 12 thì model mới hiểu đúng tốc độ hành động
    TARGET_FPS = 12 
    frame_duration = 1.0 / TARGET_FPS
    prev_inference_time = 0
    
    # Bộ nhớ đệm kết quả (Lấy trung bình 5 frame gần nhất)
    # Giúp score không bị nhảy lung tung (0.1 -> 0.9 -> 0.2)
    score_buffer = deque(maxlen=5) 
    
    # Auto Reset
    frame_count = 0
    RESET_INTERVAL = 3000 # ~4 phút reset 1 lần
    
    while True:
        if r.get(f"stop_signal_{camera_id}"):
            break
            
        # --- KỸ THUẬT XỬ LÝ RTSP CHUẨN ---
        # Luôn đọc frame để xóa buffer của camera, tránh bị delay hình ảnh
        ret, frame = cap.read()
        if not ret:
            print(f"[{camera_id}] Mất tín hiệu, kết nối lại...")
            time.sleep(2)
            cap = cv2.VideoCapture(rtsp_url)
            states = get_clean_state()
            score_buffer.clear()
            continue
            
        now = time.time()
        
        # Chỉ chạy Inference đúng nhịp 12 FPS
        # Nếu chưa đến giờ thì bỏ qua (nhưng đã read frame ở trên để ko bị cũ hình)
        if now - prev_inference_time < frame_duration:
            continue
            
        prev_inference_time = now
        
        # --- LOGIC XỬ LÝ ---
        frame_count += 1
        
        # Reset state định kỳ để tránh tràn số
        if frame_count >= RESET_INTERVAL:
            states = get_clean_state()
            score_buffer.clear()
            frame_count = 0
            gc.collect()
            
        try:
            # Preprocess
            inp = preprocess_frame(frame)
            inputs = {'image': inp}
            
            # Inference
            logits, states = run_inference_step(model, inputs, states)
            
            probs = tf.nn.softmax(logits)
            raw_prob = float(probs[0][0]) # Score thô
            
            # --- SMOOTHING (LÀM MỊN KẾT QUẢ) ---
            score_buffer.append(raw_prob)
            # Tính trung bình các score trong buffer
            avg_prob = sum(score_buffer) / len(score_buffer)
            
            # --- NÉN ẢNH GỬI VỀ CLIENT ---
            # Resize nhỏ để gửi qua mạng cho nhanh (tùy chọn)
            frame_view = cv2.resize(frame, (640, 360))
            _, buffer = cv2.imencode('.jpg', frame_view)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # Ghi Redis
            result = {
                "camera_id": camera_id,
                "fight_prob": avg_prob, # Dùng điểm trung bình
                "raw_prob": raw_prob,   # Điểm tức thời (để debug)
                "is_violent": avg_prob > 0.65, # Ngưỡng quyết định
                "timestamp": now,
                "image_base64": jpg_as_text
            }
            
            r.setex(f"cam_status_{camera_id}", 5, json.dumps(result))
            
            # Debug log trên server nếu nghi ngờ có đánh nhau
            if avg_prob > 0.6:
                print(f"[{camera_id}] Violence: {avg_prob:.2f} (Raw: {raw_prob:.2f})")

        except Exception as e:
            print(f"Error: {e}")
            states = get_clean_state()
            score_buffer.clear()

    cap.release()
    r.delete(f"cam_status_{camera_id}")
    print(f"[{camera_id}] Stopped.")