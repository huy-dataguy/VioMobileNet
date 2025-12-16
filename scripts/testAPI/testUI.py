import requests
import cv2
import numpy as np
import base64
import time
import sys

# ================= CẤU HÌNH =================
# Thay đổi ID camera cho phù hợp với cái bạn đang start
CAMERA_ID = "cam01" 
API_URL = f"http://192.168.1.5:8000/camera/status/{CAMERA_ID}"

# Kích thước màn hình hiển thị mong muốn (Phóng to từ ảnh gốc 320x180)
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

def draw_text_with_outline(img, text, x, y, font_scale=0.8, color=(255, 255, 255), thickness=2):
    """Hàm vẽ chữ có viền đen để dễ đọc trên mọi nền"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2) # Viền đen
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)         # Chữ màu

def monitor_dashboard():
    print(f"Đang kết nối Dashboard tới: {API_URL}")
    print("Nhấn 'q' để thoát.")
    
    # Biến tính Client FPS
    prev_frame_time = 0
    
    while True:
        try:
            # 1. Gọi API
            try:
                response = requests.get(API_URL, timeout=1)
                if response.status_code != 200:
                    print(f"Server Error: {response.status_code}")
                    time.sleep(1)
                    continue
                data = response.json()
            except requests.exceptions.ConnectionError:
                print("Mất kết nối tới Server...")
                time.sleep(1)
                continue

            # Kiểm tra trạng thái offline
            if data.get("status") == "offline":
                # Tạo màn hình đen thông báo offline
                blank_screen = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
                cv2.putText(blank_screen, f"CAMERA {CAMERA_ID} OFFLINE", (100, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                cv2.imshow("Urban Safety Monitor", blank_screen)
                if cv2.waitKey(100) & 0xFF == ord('q'): break
                continue

            # 2. Giải mã hình ảnh (Key mới là 'image_preview')
            img_str = data.get("image_preview")
            if not img_str:
                continue
                
            img_bytes = base64.b64decode(img_str)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame_small = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Phóng to ảnh để dễ nhìn (Upscale)
            frame = cv2.resize(frame_small, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)

            # 3. Lấy các chỉ số từ JSON mới
            is_violent = data.get("is_violent", False)
            score = data.get("score", 0.0)
            server_fps = data.get("fps", 0.0)
            latency = data.get("latency_ms", 0.0)
            evidence_url = data.get("evidence_url") # Link MinIO
            
            # --- VẼ GIAO DIỆN DASHBOARD ---
            
            # Màu sắc chủ đạo
            theme_color = (0, 0, 255) if is_violent else (0, 255, 0) # Đỏ hoặc Xanh
            status_text = "VIOLENCE DETECTED" if is_violent else "NORMAL STATE"
            
            # A. Vẽ khung viền cảnh báo
            if is_violent:
                cv2.rectangle(frame, (0, 0), (DISPLAY_WIDTH, DISPLAY_HEIGHT), theme_color, 15)
            
            # B. Header: Thông tin Camera & Trạng thái
            # Vẽ nền mờ cho header
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (DISPLAY_WIDTH, 60), (0,0,0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            draw_text_with_outline(frame, f"CAM: {CAMERA_ID}", 20, 40, 1.0, (255, 255, 255), 2)
            draw_text_with_outline(frame, status_text, 250, 40, 1.0, theme_color, 3)

            # C. Footer: Các chỉ số kỹ thuật (Góc phải dưới)
            info_y = DISPLAY_HEIGHT - 20
            draw_text_with_outline(frame, f"Server FPS: {server_fps}", DISPLAY_WIDTH - 250, info_y - 60, 0.6, (200, 200, 200), 1)
            draw_text_with_outline(frame, f"Latency: {latency}ms", DISPLAY_WIDTH - 250, info_y - 30, 0.6, (200, 200, 200), 1)
            
            # Tính Client FPS (Tốc độ vẽ của máy bạn)
            new_frame_time = time.time()
            client_fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time
            draw_text_with_outline(frame, f"Client FPS: {client_fps:.1f}", DISPLAY_WIDTH - 250, info_y, 0.6, (150, 150, 150), 1)

            # D. Thanh điểm số (Score Bar - Góc trái dưới)
            bar_width = 300
            bar_height = 20
            bar_x = 20
            bar_y = DISPLAY_HEIGHT - 40
            
            # Vẽ khung thanh
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 2)
            # Vẽ mức điểm hiện tại
            fill_width = int(score * bar_width)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), theme_color, -1)
            draw_text_with_outline(frame, f"Violence Score: {score:.4f}", bar_x, bar_y - 10, 0.7, (255, 255, 255), 2)

            # E. Chỉ báo MinIO (Nếu đã lưu bằng chứng)
            if evidence_url:
                cv2.putText(frame, "REC [EVIDENCE SAVED]", (DISPLAY_WIDTH - 350, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                # Chấm đỏ nhấp nháy giả lập
                if int(time.time() * 2) % 2 == 0:
                    cv2.circle(frame, (DISPLAY_WIDTH - 370, 110), 10, (0, 0, 255), -1)

            # 4. Show cửa sổ
            cv2.imshow("Urban Safety Monitor", frame)

            # Nhấn 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        except Exception as e:
            print(f"Lỗi Client: {e}")
            time.sleep(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor_dashboard()
