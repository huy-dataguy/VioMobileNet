import os
import cv2
import numpy as np
import tensorflow as tf
from celery import Celery
from official.projects.movinet.modeling import movinet, movinet_model

# 1. Cấu hình Celery
celery_app = Celery('video_tasks', broker='redis://redis:6379/0', backend='redis://redis:6379/0')

# 2. Cấu hình Model
# Lưu ý: Đảm bảo đường dẫn weights đúng trong Docker (ví dụ: /app/weights/...)
CHECKPOINT_DIR = './models/movinet_a3_12fps_64bs_0.001lr_0.3dr_0tl/'
MODEL_ID = 'a3'
TARGET_FPS = 12
RESOLUTION = 256

# --- PHẦN BUILD MODEL ---
def build_model_optimized():
    print(f"Building MoViNet {MODEL_ID}...")
    backbone = movinet.Movinet(
        model_id=MODEL_ID,
        causal=True,
        conv_type='2plus1d',
        se_type='2plus3d',
        activation='hard_swish',
        gating_activation='hard_sigmoid',
        use_positional_encoding=True,
        use_external_states=True,
    )

    model = movinet_model.MovinetClassifier(
        backbone, num_classes=2, output_states=True)

    # Build input giả
    inputs = tf.ones([1, 1, RESOLUTION, RESOLUTION, 3])
    model.build(inputs.shape)

    # Load weights
    print(f"Loading weights from: {CHECKPOINT_DIR}")
    model.load_weights(tf.train.latest_checkpoint(CHECKPOINT_DIR)).expect_partial()
    
    return model

# --- KHỞI TẠO GLOBAL (Chạy 1 lần khi Worker start) ---
print("--- WORKER STARTING ---")
inference_model = build_model_optimized()
# Tạo state rỗng ban đầu để copy dùng dần
TEMPLATE_STATES = inference_model.init_states(tf.shape(tf.ones([1, 1, RESOLUTION, RESOLUTION, 3])))
print("--- MODEL READY ON GPU A4000 ---")

# --- HÀM XỬ LÝ VIDEO (OPENCV) ---
def load_video_smart_sampling(path, target_fps=12, target_size=256):
    frames = []
    cap = cv2.VideoCapture(path)
    
    # Lấy FPS gốc của video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0: original_fps = 30 # Fallback nếu không đọc được
    
    # Tính bước nhảy (Ví dụ: Video 24fps, Target 12fps => Lấy mỗi frame thứ 2)
    step = max(1, int(round(original_fps / target_fps)))
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Chỉ lấy frame nếu chia hết cho step
        if count % step == 0:
            # Resize & Normalize
            frame = cv2.resize(frame, (target_size, target_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0
            frames.append(frame)
            
        count += 1
        
    cap.release()
    
    # Convert sang Tensor: (num_frames, H, W, 3)
    if len(frames) == 0: return None
    return tf.constant(np.array(frames), dtype=tf.float32)

# Để tăng tốc, wrap bước inference vào tf.function
@tf.function
def run_inference_step(inputs, states):
    outputs = inference_model({**inputs, **states})
    return outputs[0], outputs[1]

# --- CELERY TASK ---
@celery_app.task(bind=True)
def predict_violence(self, video_path):
    try:
        self.update_state(state='PROCESSING')
        print(f"Processing video: {video_path}")
        
        # 1. Đọc video (có nhảy frame)
        video_tensor = load_video_smart_sampling(video_path, target_fps=TARGET_FPS, target_size=RESOLUTION)
        
        if video_tensor is None:
            return {"status": "error", "message": "Cannot read video or video is empty"}

        # 2. Inference
        # Reset state về rỗng cho video mới
        states = TEMPLATE_STATES
        logits = None
        
        num_frames = video_tensor.shape[0]
        print(f"Running inference on {num_frames} frames...")

        for i in range(num_frames):
            # Tạo batch (1, 1, H, W, 3)
            frame = video_tensor[i][tf.newaxis, tf.newaxis, ...]
            
            # Chạy model
            inputs = {'image': frame}
            logits, states = run_inference_step(inputs, states)
        
        # 3. Kết quả cuối cùng
        probs = tf.nn.softmax(logits)
        fight_prob = float(probs[0][0])
        no_fight_prob = float(probs[0][1])
        
        print(f"Done! Fight Prob: {fight_prob:.4f}")

        # 4. Xóa file sau khi xong
        if os.path.exists(video_path):
            os.remove(video_path)
            
        return {
            "status": "success",
            "is_violent": fight_prob > 0.5,
            "score": fight_prob,
            "details": {"fight": fight_prob, "normal": no_fight_prob}
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        # Cố gắng xóa file nếu lỗi
        if os.path.exists(video_path):
            os.remove(video_path)
        return {"status": "error", "message": str(e)}
