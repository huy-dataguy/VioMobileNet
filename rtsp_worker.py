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
    print(f"--- [RTSP WORKER {camera_id}] STARTING FINAL PRODUCTION ---")
    
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
    cap = cv2.VideoCapture(rtsp_url)
    
    # --- CẤU HÌNH ---
    TARGET_FPS = 12 
    frame_duration = 1.0 / TARGET_FPS
    prev_inference_time = 0
    score_buffer = deque(maxlen=5) 
    
    # Cấu hình Scene Detect (Chống nhiễu khi rung lắc/che cam)
    prev_frame_gray = None
    SCENE_CHANGE_THRESHOLD = 35.0 # Ngưỡng phát hiện thay đổi đột ngột
    
    frame_count = 0
    RESET_INTERVAL = 3000
    
    while True:
        if r.get(f"stop_signal_{camera_id}"):
            break
            
        ret, frame = cap.read()
        if not ret:
            print(f"[{camera_id}] Mất tín hiệu, thử lại...")
            time.sleep(2)
            cap = cv2.VideoCapture(rtsp_url)
            states = get_clean_state()
            score_buffer.clear()
            prev_frame_gray = None
            continue
            
        now = time.time()
        # Chỉ chạy đúng 12 FPS
        if now - prev_inference_time < frame_duration:
            continue
        prev_inference_time = now
        
        # --- 1. PHÁT HIỆN SỐC HÌNH ẢNH (SCENE CHANGE) ---
        reset_trigger = False
        try:
            # Resize siêu nhỏ để so sánh nhanh
            small_frame = cv2.resize(frame, (64, 64))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame_gray is not None:
                score_diff = cv2.absdiff(gray, prev_frame_gray)
                mean_diff = np.mean(score_diff)
                
                # Nếu thay đổi quá lớn (che cam, quay cam đi chỗ khác)
                if mean_diff > SCENE_CHANGE_THRESHOLD:
                    print(f"[{camera_id}] ⚡ Đổi cảnh đột ngột (Diff: {mean_diff:.1f}) -> Reset Memory")
                    states = get_clean_state() # Xóa ký ức cũ
                    score_buffer.clear()
                    reset_trigger = True
            
            prev_frame_gray = gray
        except: pass
        # ------------------------------------------------

        frame_count += 1
        if frame_count >= RESET_INTERVAL and not reset_trigger:
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
            raw_prob = float(probs[0][0])
            
            # --- SMOOTHING ---
            # Nếu vừa bị reset do sốc hình ảnh, khoan hãy tin kết quả ngay
            if not reset_trigger:
                score_buffer.append(raw_prob)
            
            avg_prob = 0.0
            if len(score_buffer) > 0:
                avg_prob = sum(score_buffer) / len(score_buffer)
            else:
                avg_prob = raw_prob # Fallback nếu buffer rỗng

            # --- GỬI VỀ CLIENT ---
            frame_view = cv2.resize(frame, (480, 270)) # Resize nhỏ hơn chút nữa cho nhanh
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # Giảm chất lượng xuống 50%
            _, buffer = cv2.imencode('.jpg', frame_view, encode_param)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            result = {
                "camera_id": camera_id,
                "fight_prob": avg_prob,
                "raw_prob": raw_prob,
                "is_violent": avg_prob > 0.7, # Ngưỡng báo động
                "timestamp": now,
                "image_base64": jpg_as_text
            }
            
            r.setex(f"cam_status_{camera_id}", 5, json.dumps(result))
            
            if avg_prob > 0.6:
                print(f"[{camera_id}] Violence: {avg_prob:.2f}")

        except Exception as e:
            print(f"Error: {e}")
            states = get_clean_state()
            score_buffer.clear()

    cap.release()
    r.delete(f"cam_status_{camera_id}")
    print(f"[{camera_id}] Stopped.")