import time
import redis
import json
import cv2
import tensorflow as tf
import gc
from core import build_model_optimized, preprocess_frame, run_inference_step

r = redis.Redis(host='redis', port=6379, db=0)

def run_camera_process(camera_id, rtsp_url):
    print(f"--- [RTSP WORKER {camera_id}] STARTING ---")
    
    # Cấu hình GPU Memory Growth để tránh chiếm hết VRAM
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except: pass

    # Load Model riêng cho process này
    model = build_model_optimized()
    
    def get_clean_state():
        return model.init_states(tf.shape(tf.ones([1, 1, 256, 256, 3])))

    states = get_clean_state()
    cap = cv2.VideoCapture(rtsp_url)
    
    prev_time = 0
    fps_limit = 10  # Giới hạn FPS xử lý để tiết kiệm GPU
    frame_count = 0
    RESET_INTERVAL = 5000 # Reset state sau mỗi 5000 frames (~8 phút)
    
    while True:
        # 1. Kiểm tra tín hiệu dừng từ Redis
        if r.get(f"stop_signal_{camera_id}"):
            print(f"[{camera_id}] Stop signal received.")
            break
            
        ret, frame = cap.read()
        if not ret:
            print(f"[{camera_id}] Lost signal, retrying...")
            time.sleep(2)
            cap = cv2.VideoCapture(rtsp_url)
            states = get_clean_state() # Reset state khi mất tín hiệu
            continue
            
        # 2. Kiểm soát FPS
        now = time.time()
        if now - prev_time < 1./fps_limit:
            continue
        prev_time = now
        
        # 3. Auto Reset State (Tránh lỗi tích lũy sai số)
        frame_count += 1
        if frame_count >= RESET_INTERVAL:
            states = get_clean_state()
            frame_count = 0
            gc.collect() # Dọn rác RAM
            
        # 4. Inference
        try:
            inp = preprocess_frame(frame)
            inputs = {'image': inp}
            
            # Gọi hàm tf.function từ core
            logits, states = run_inference_step(model, inputs, states)
            
            probs = tf.nn.softmax(logits)
            fight_prob = float(probs[0][0])
            
            # 5. Ghi kết quả vào Redis (API sẽ đọc từ đây)
            result = {
                "camera_id": camera_id,
                "fight_prob": fight_prob,
                "is_violent": fight_prob > 0.7,
                "timestamp": now
            }
            # Key tồn tại trong 5 giây
            r.setex(f"cam_status_{camera_id}", 5, json.dumps(result))
            
            if fight_prob > 0.8:
                print(f"!!! VIOLENCE DETECTED [{camera_id}]: {fight_prob:.2f}")

        except Exception as e:
            print(f"Error inference: {e}")
            states = get_clean_state()

    cap.release()
    r.delete(f"cam_status_{camera_id}")
    print(f"[{camera_id}] Stopped.")