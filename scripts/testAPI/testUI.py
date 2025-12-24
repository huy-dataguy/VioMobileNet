import requests
import cv2
import numpy as np
import base64
import time
import sys

# Cấu hình danh sách camera
CAM_LIST = [f"cam{i:02d}" for i in range(1, 9)]  # ['cam01', 'cam02', ..., 'cam08']
BASE_URL = "http://localhost:8000/camera/status/"

def monitor_gui():
    current_cam_idx = 0  # Mặc định bắt đầu với cam01
    print(f"🚀 Đang khởi tạo màn hình giám sát (OpenCV)...")
    print(f"🎮 HƯỚNG DẪN: Nhấn phím [1-8] để chuyển camera, [Q] để thoát.")
    
    session = requests.Session()
    
    # Kích thước hiển thị 16:9
    DISPLAY_WIDTH = 800
    DISPLAY_HEIGHT = 450 

    while True:
        camera_id = CAM_LIST[current_cam_idx]
        url = f"{BASE_URL}{camera_id}"
        
        try:
            start_time = time.time()
            response = session.get(url, timeout=1)
            data = response.json()
            
            # Xử lý khi camera offline
            if data.get("status") == "offline":
                # Tạo khung đen thông báo offline thay vì đóng cửa sổ
                frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, f"CAMERA {camera_id.upper()} OFFLINE", (150, 225), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Press [1-8] to switch", (250, 270), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            else:
                img_str = data.get("image_preview", "") 
                if not img_str:
                    continue
                    
                img_bytes = base64.b64decode(img_str)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                raw_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                # Căn chỉnh tỷ lệ
                frame = cv2.resize(raw_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_CUBIC)

                # Vẽ thông tin
                prob = data.get("score", 0.0) 
                is_violent = data.get("is_violent", False)
                server_label = data.get("server", "N/A")
                
                color = (0, 0, 255) if is_violent else (0, 255, 0)
                text_status = "VIOLENCE DETECTED!" if is_violent else "NORMAL"
                
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 6)
                
                # Header thông tin
                cv2.putText(frame, f"MONITORING: {camera_id.upper()} ({server_label})", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"STATUS: {text_status}", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(frame, f"SCORE: {prob:.4f}", (20, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # Hướng dẫn nhanh ở góc dưới
                cv2.putText(frame, "Keys: [1-8] Switch Cam | [Q] Quit", (20, DISPLAY_HEIGHT - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                fps_show = 1.0 / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {fps_show:.1f}", (DISPLAY_WIDTH - 100, DISPLAY_HEIGHT - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("MoViNet AI Monitor - Realtime", frame)

            # BẮT PHÍM ĐỂ CHUYỂN CAMERA
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # Phím 1 tương ứng code 49, phím 8 tương ứng 56
            elif ord('1') <= key <= ord('8'):
                current_cam_idx = key - ord('1')
                print(f"🔄 Đang chuyển sang: {CAM_LIST[current_cam_idx].upper()}")
        
        except Exception as e:
            print(f"Lỗi kết nối hoặc xử lý: {e}")
            time.sleep(0.5)

    session.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor_gui()