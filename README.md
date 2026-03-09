# VioMobileNet - Real-time AI Violence Monitoring System

A high-performance system designed to analyze violent behavior from RTSP camera streams and video uploads. Powered by **MoViNet**, optimized for ultra-high throughput and minimal latency.

## Key Features

* **Dual Inference Pipeline:**
* Operates two parallel inference processes: `HIGH_SERVER` and `LOW_MED_SERVER`.
* **Prioritized Queuing:** High-risk (`high`) cameras are isolated in a dedicated stream to ensure the highest possible monitoring frequency.


* **Instant Score Response:**
* **Boost Logic:** Uses an exponential function to spike scores when the threshold exceeds **0.2**, triggering alarms the moment violence begins.
* **Quick Drop Logic:** Force-resets scores to **0** as soon as the scene stabilizes, eliminating the "accumulation lag" common in AI temporal analysis.


* **LIFO (Last-In-First-Out) & Interleaving:**
* **Frame Interleaving:** Distributes frame processing across cameras to prevent any single stream from hogging the queue.
* **Aggressive LIFO:** Automatically purges stale frames to ensure system-wide latency stays **under 100ms**.


* **Non-blocking Background Upload:**
* Evidence images are uploaded to **MinIO** via dedicated background threads. This prevents the primary AI inference loop from stuttering during data transmission.


* **Smart Redis Caching:**
* Dynamic TTL (Time-to-Live) management (up to **60 seconds**) for violent events, ensuring API/CURL queries receive stable, reliable data.



---

## Directory Structure

```text
VioMobileNet/
├── app.py              # API Gateway: Routes cameras to the correct queue (High vs. Low/Med).
├── core.py             # AI Core: GPU Memory Growth and TensorFlow thread management.
├── rtsp_worker.py      # The Brain: Parallel inference logic and multi-cadence streaming.
├── worker.py           # Celery Worker: Offline video processing for file uploads.
├── requirements.txt    # Dependencies: TensorFlow 2.15, OpenCV, Redis, MinIO...
└── docker-compose.yml  # Orchestration: Links API, Workers, Redis, and MinIO.

```

---

## Operational Workflow

1. **Ingestion:** The Streamer reads RTSP feeds, resizes frames to 256x256, and pushes them into risk-based queues using interleaving.
2. **Dual Inference:**
* `HIGH_SERVER` processes priority cameras at high FPS.
* `LOW_MED_SERVER` handles standard monitoring to conserve resources.


3. **Score Processing:** Applies **Max Pooling** and exponential scaling to make the model "sensory-sharp" to aggressive actions.
4. **Result Distribution:**
* Status data is cached in **Redis** with flexible TTL.
* Visual evidence is offloaded via **Threaded Upload** to **MinIO**.



---

## Getting Started

### 1. System Requirements

* **GPU:** NVIDIA RTX 30 Series or higher (CUDA 12.x support).
* **RAM:** 16GB Minimum (System consumes ~3GB for two inference pipelines).
* **Environment:** Docker & NVIDIA Container Toolkit installed.

### 2. Deployment

```bash
# Spin up the entire stack
docker-compose up -d --build

# Monitor logs for parallel inference streams
docker logs -f viomobilenet_api

```

### 3. UI Controls (Monitoring Client)

Toggle between camera feeds instantly using keyboard shortcuts:

* **Keys [1-8]:** Switch between Camera 01 and Camera 08.
* **Key [Q]:** Exit the monitoring system.

---

## Performance Benchmarks (Tested on RTX 4060 Ti)

* **AI Inference Speed:** ~30ms - 50ms per frame.
* **Capacity:** Supports 8-10 simultaneous streams with high stability.
* **Pipeline Latency:** Consistently maintained **< 100ms** via LIFO and Background Uploading.
