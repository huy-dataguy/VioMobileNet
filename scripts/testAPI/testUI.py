import requests
import cv2
import numpy as np
import base64
import time
import sys

# IP VPS
# URL = "http://192.168.0.200:8000/camera/status/cam1"
URL = "http://192.168.0.200:8000/camera/status/cam01"
def monitor_gui():
    print("🚀 Đang khởi tạo màn hình giám sát (OpenCV)...")
    
    while True:
        try:
            start_time = time.time()
            
            # 1. Gọi API lấy dữ liệu
            response = requests.get(URL, timeout=2)
            data = response.json()
            
            if data.get("status") == "offline":
                print("Camera Offline...")
                time.sleep(1)
                continue

            # 2. Giải mã hình ảnh từ Base64
            img_str = data.get("image_base64", "")
            if not img_str:
                continue
                
            img_bytes = base64.b64decode(img_str)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 3. Vẽ thông tin lên hình ảnh
            prob = data.get("fight_prob", 0.0)
            is_violent = data.get("is_violent", False)
            camera_id = data.get("camera_id", "Unknown")
            
            # Màu sắc: Xanh lá (Bình thường) hoặc Đỏ (Bạo lực)
            color = (0, 0, 255) if is_violent else (0, 255, 0) # BGR format
            text_status = "VIOLENCE DETECTED!" if is_violent else "NORMAL"
            
            # Vẽ viền
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 10)
            
            # Vẽ thông số
            cv2.putText(frame, f"CAM: {camera_id}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(frame, f"STATUS: {text_status}", (30, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            cv2.putText(frame, f"SCORE: {prob:.4f}", (30, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Tính FPS hiển thị (client side fps)
            fps_show = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"Client FPS: {fps_show:.1f}", (30, frame.shape[0] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # 4. Show cửa sổ
            cv2.imshow("MoViNet AI Monitor - Realtime", frame)

            # Nhấn 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        except Exception as e:
            print(f"Lỗi: {e}")
            time.sleep(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor_gui()
