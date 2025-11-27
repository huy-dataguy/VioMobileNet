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
    print(f"--- [RTSP WORKER {camera_id}] STARTING WITH SCENE DETECT ---")
    
    # 1. Config GPU
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except: pass

    model = build_model_optimized()
    
    def get_clean_state():
        return model.init_states(tf.shape(tf.ones([1, 1, 256, 256, 3])))

    states = get_clean_state()
    cap = cv2.VideoCapture(rtsp_url)
    
    TARGET_FPS = 12 
    frame_duration = 1.0 / TARGET_FPS
    prev_inference_time = 0
    score_buffer = deque(maxlen=5) 
    
    # Biến để so sánh Scene Change
    prev_frame_gray = None 
    SCENE_CHANGE_THRESHOLD = 30.0 # Ngưỡng thay đổi pixel (càng nhỏ càng nhạy)
    
    frame_count = 0
    RESET_INTERVAL = 3000
    
    while True:
        if r.get(f"stop_signal_{camera_id}"):
            break
            
        ret, frame = cap.read()
        if not ret:
            print(f"[{camera_id}] Lost signal...")
            time.sleep(2)
            cap = cv2.VideoCapture(rtsp_url)
            states = get_clean_state()
            continue
            
        now = time.time()
        if now - prev_inference_time < frame_duration:
            continue
        prev_inference_time = now
        
        # --- THUẬT TOÁN PHÁT HIỆN CHUYỂN CẢNH (SCENE CUT) ---
        # 1. Resize nhỏ để tính toán cho nhanh
        small_frame = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        reset_trigger = False
        if prev_frame_gray is not None:
            # Tính độ sai lệch giữa frame hiện tại và frame trước
            score_diff = cv2.absdiff(gray, prev_frame_gray)
            mean_diff = np.mean(score_diff)
            
            # Nếu khác biệt quá lớn -> Có sự chuyển cảnh (Video này qua Video kia)
            if mean_diff > SCENE_CHANGE_THRESHOLD:
                print(f"[{camera_id}] 🎬 SCENE CHANGE DETECTED (Diff: {mean_diff:.2f}) -> RESET STATE")
                states = get_clean_state() # Xóa ký ức cũ ngay lập tức
                score_buffer.clear()       # Xóa buffer điểm số cũ
                reset_trigger = True
        
        prev_frame_gray = gray
        # -------------------------------------------------------

        frame_count += 1
        # Reset định kỳ (nếu không có scene change)
        if frame_count >= RESET_INTERVAL and not reset_trigger:
            states = get_clean_state()
            score_buffer.clear()
            frame_count = 0
            gc.collect()
            
        try:
            inp = preprocess_frame(frame)
            inputs = {'image': inp}
            logits, states = run_inference_step(model, inputs, states)
            
            probs = tf.nn.softmax(logits)
            raw_prob = float(probs[0][0])
            
            # Nếu vừa reset scene, đừng vội nạp vào buffer ngay, đợi 1-2 frame ổn định
            if not reset_trigger:
                score_buffer.append(raw_prob)
            
            if len(score_buffer) > 0:
                avg_prob = sum(score_buffer) / len(score_buffer)
            else:
                avg_prob = raw_prob

            # --- Gửi ảnh về (như cũ) ---
            height, width = frame.shape[:2]
            new_width = 480
            new_height = int(height * (new_width / width))
            frame_small = cv2.resize(frame, (new_width, new_height))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            _, buffer = cv2.imencode('.jpg', frame_small, encode_param)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            result = {
                "camera_id": camera_id,
                "fight_prob": avg_prob,
                "is_violent": avg_prob > 0.7,
                "timestamp": now,
                "image_base64": jpg_as_text
            }
            
            r.setex(f"cam_status_{camera_id}", 5, json.dumps(result))
            
            if avg_prob > 0.6:
                print(f"[{camera_id}] Violence: {avg_prob:.2f}")

        except Exception as e:
            print(f"Error: {e}")
            states = get_clean_state()

    cap.release()
    r.delete(f"cam_status_{camera_id}")
    print(f"[{camera_id}] Stopped.")