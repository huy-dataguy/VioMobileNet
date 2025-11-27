Hệ thống này sẽ hoạt động như sau:

API Upload Video: Dùng cơ chế Queue (Celery) để xếp hàng xử lý từng video một, tránh quá tải.

API RTSP Camera: Dùng cơ chế Multiprocessing để tách riêng mỗi camera ra một tiến trình độc lập, chạy song song với API chính.

Tối ưu GPU: Cả hai luồng đều dùng chung logic Model MoViNet đã tối ưu, chia sẻ tài nguyên GPU hợp lý.

VioMobileNet/
├── app.py              # API Gateway (Quản lý Upload & Camera)
├── core.py             # Logic AI (Load model, Inference) - Dùng chung
├── worker.py           # Worker cho Video Upload (Celery)
├── rtsp_worker.py      # Worker cho RTSP Camera (Multiprocessing)
├── requirements.txt    # Thư viện
├── Dockerfile          # Cấu hình build
└── docker-compose.yml  # Cấu hình chạy toàn bộ hệ thống