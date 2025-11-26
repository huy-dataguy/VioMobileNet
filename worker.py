import os
import cv2
import numpy as np
import tensorflow as tf
from celery import Celery
from core import build_model_optimized, get_template_states, run_inference_step, RESOLUTION

# Cấu hình Celery
celery_app = Celery('video_tasks', broker='redis://redis:6379/0', backend='redis://redis:6379/0')

# Khởi tạo Model Global (Chỉ chạy 1 lần khi worker start)
print("--- [UPLOAD WORKER] STARTING ---")
inference_model = build_model_optimized()
TEMPLATE_STATES = get_template_states(inference_model)
print("--- [UPLOAD WORKER] READY ON GPU ---")

def load_video_smart_sampling(path, target_fps=12, target_size=256):
    frames = []
    cap = cv2.VideoCapture(path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0: original_fps = 30
    step = max(1, int(round(original_fps / target_fps)))
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if count % step == 0:
            frame = cv2.resize(frame, (target_size, target_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0
            frames.append(frame)
        count += 1
    cap.release()
    
    if len(frames) == 0: return None
    return tf.constant(np.array(frames), dtype=tf.float32)

@celery_app.task(bind=True, name='predict_violence')
def predict_violence(self, video_path):
    try:
        self.update_state(state='PROCESSING')
        print(f"Processing video: {video_path}")
        
        video_tensor = load_video_smart_sampling(video_path, target_fps=12, target_size=RESOLUTION)
        if video_tensor is None:
            return {"status": "error", "message": "Cannot read video"}

        states = TEMPLATE_STATES
        logits = None
        num_frames = video_tensor.shape[0]
        
        print(f"Inference on {num_frames} frames...")
        for i in range(num_frames):
            frame = video_tensor[i][tf.newaxis, tf.newaxis, ...]
            inputs = {'image': frame}
            logits, states = run_inference_step(inference_model, inputs, states)
        
        probs = tf.nn.softmax(logits)
        fight_prob = float(probs[0][0])
        
        if os.path.exists(video_path): os.remove(video_path)
        
        return {
            "status": "success",
            "is_violent": fight_prob > 0.5,
            "score": fight_prob
        }
    except Exception as e:
        print(f"Error: {e}")
        if os.path.exists(video_path): os.remove(video_path)
        return {"status": "error", "message": str(e)}