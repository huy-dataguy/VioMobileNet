import os
import cv2
import numpy as np
import tensorflow as tf
from official.projects.movinet.modeling import movinet, movinet_model

# Cấu hình Model
CHECKPOINT_DIR = './models/movinet_a3_12fps_64bs_0.001lr_0.3dr_0tl/'
MODEL_ID = 'a3'
RESOLUTION = 256

def build_model_optimized():
    """Hàm load model và weights vào GPU"""
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
    # Fallback an toàn nếu không tìm thấy file
    if os.path.exists(CHECKPOINT_DIR):
        latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        if latest:
             model.load_weights(latest).expect_partial()
        else:
             print("WARNING: No checkpoint found in dir, using random weights!")
    else:
        print(f"ERROR: Checkpoint dir not found: {CHECKPOINT_DIR}")
    
    return model

def get_template_states(model):
    """Tạo state rỗng ban đầu"""
    return model.init_states(tf.shape(tf.ones([1, 1, RESOLUTION, RESOLUTION, 3])))

def preprocess_frame(frame, target_size=256):
    """Chuẩn hóa 1 frame ảnh"""
    frame = cv2.resize(frame, (target_size, target_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    # Thêm batch dimension: (1, 1, H, W, 3)
    return tf.constant(frame[np.newaxis, np.newaxis, ...], dtype=tf.float32)

# Wrap tf.function để tăng tốc inference
@tf.function
def run_inference_step(model, inputs, states):
    outputs = model({**inputs, **states})
    return outputs[0], outputs[1]