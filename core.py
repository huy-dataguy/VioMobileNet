import os
import cv2
import numpy as np

import tensorflow as tf
from official.projects.movinet.modeling import movinet, movinet_model

# Cấu hình Model
CHECKPOINT_DIR = './models/movinet_a3_12fps_64bs_0.001lr_0.3dr_0tl/'
MODEL_ID = 'a3'
RESOLUTION = 256

def setup_gpu_config():
    """Hàm này phải được gọi ĐẦU TIÊN trong mỗi Process"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU Configured: Memory Growth = True")
        except RuntimeError as e:
            print(f"GPU Config Error: {e}")
    # Ép mỗi tiến trình chỉ dùng 1 luồng, tránh xung đột CPU gây crash
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

def build_model_optimized():
    """Hàm load model và weights"""
    setup_gpu_config()
    
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

    inputs = tf.ones([1, 1, RESOLUTION, RESOLUTION, 3])
    model.build(inputs.shape)

    if os.path.exists(CHECKPOINT_DIR):
        latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        if latest:
             model.load_weights(latest).expect_partial()
    else:
        print(f"ERROR: Checkpoint dir not found: {CHECKPOINT_DIR}")
    
    return model

def get_template_states(model):
    return model.init_states(tf.shape(tf.ones([1, 1, RESOLUTION, RESOLUTION, 3])))

def preprocess_frame(frame, target_size=256):
    frame = cv2.resize(frame, (target_size, target_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    return tf.constant(frame[np.newaxis, np.newaxis, ...], dtype=tf.float32)

@tf.function
def run_inference_step(model, inputs, states):
    outputs = model({**inputs, **states})
    return outputs[0], outputs[1]