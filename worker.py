# worker.py
import os
from celery import Celery
import tensorflow as tf
# Import các hàm build_streaming_model, video_to_gif_tensor, streaming_inference của bạn ở đây
# ... (Copy các hàm phụ trợ vào file này hoặc import từ module utils)

# Cấu hình Celery kết nối với Redis
celery_app = Celery('video_tasks', broker='redis://redis:6379/0', backend='redis://redis:6379/0')

# LOAD MODEL GLOBAL (Chỉ chạy 1 lần khi worker khởi động)
print("Loading MoViNet Model...")
inference_model, RESOLUTION, FRAMES = build_streaming_model() # Hàm của bạn
initial_input_shape = tf.shape(tf.ones([1, 1, RESOLUTION, RESOLUTION, 3]))
init_states = inference_model.init_states(initial_input_shape)
print("Model Loaded!")

@celery_app.task(bind=True)
def predict_violence(self, video_path):
    try:
        # Update trạng thái
        self.update_state(state='PROCESSING')
        
        # 1. Preprocess
        tensor = video_to_gif_tensor(video_path, fps=12) # Hàm của bạn
        
        # 2. Inference
        # Lưu ý: Streaming inference của bạn đang stateful, cần reset state cho mỗi video mới
        clean_states = init_states 
        final_probs = streaming_inference(tensor, clean_states, inference_model) # Hàm của bạn
        
        # 3. Format kết quả
        result = {
            "fight_prob": float(final_probs[0][0]),
            "no_fight_prob": float(final_probs[0][1]),
            "is_violence": bool(final_probs[0][0] > 0.5) # Ngưỡng ví dụ
        }
        
        # 4. Dọn dẹp file video (tùy chọn)
        if os.path.exists(video_path):
            os.remove(video_path)
            
        return result
        
    except Exception as e:
        return {"error": str(e)}
