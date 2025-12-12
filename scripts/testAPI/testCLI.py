import requests
import time
import os
import sys
from datetime import datetime
import base64 # <--- Thêm import này ở đầu file
# IP VPS của bạn
# URL = "http://192.168.0.200:8000/camera/status/cam1"

URL = "http://192.168.0.200:8000/camera/status/liveFight"
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def monitor():
    print("Đang kết nối tới Camera Realtime... (Nhấn Ctrl+C để dừng)")
    time.sleep(1)

    try:
        while True:
            try:
                # 1. Gọi API
                response = requests.get(URL, timeout=2)
                data = response.json()
                
                clear_screen()

                status = data.get("status", "unknown")
                timestamp = data.get("timestamp", 0)
                
                # --- LOGIC MỚI: TÍNH TOÁN THỜI GIAN ---
                # Đổi timestamp sang giờ đọc được
                dt_object = datetime.fromtimestamp(timestamp)
                human_time = dt_object.strftime("%H:%M:%S.%f")[:-3] # Giờ:Phút:Giây.ms
                
                # Tính độ trễ (Latency) = Giờ máy bạn - Giờ Server trả về
                # Lưu ý: Yêu cầu đồng hồ laptop và VPS phải tương đối khớp nhau (NTP)
                now = time.time()
                latency = now - timestamp
                
                print("="*50)
                print(f"CAMERA MONITORING SYSTEM")
                print("="*50)

                if status == "offline":
                    print(f"TRẠNG THÁI: {status.upper()}")
                    print("Camera chưa bật hoặc bị lỗi kết nối.")
                else:
                    prob = data.get("fight_prob", 0.0)
                    is_violent = data.get("is_violent", False)

                    print(f"Camera ID : {data.get('camera_id')}")
                    print(f"Frame Time: {human_time} (Server)")
                    
                    # Hiển thị độ trễ bằng màu sắc
                    if latency < 1.0:
                        print(f"Latency   : {latency:.3f}s (Realtime)")
                    elif latency < 3.0:
                        print(f"Latency   : {latency:.3f}s (Chấp nhận được)")
                    else:
                        print(f"Latency   : {latency:.3f}s (Lag)")
                        
                    print("-" * 50)
                    
                    if is_violent:
                        print(f"\nCẢNH BÁO BẠO LỰC!")
                        print(f"Score: {prob:.4f}")
                        print("\n" + "█" * int(prob * 40))
                    else:
                        print(f"\nBình thường")
                        print(f"Score: {prob:.4f}")
                        print("\n" + "." * int(prob * 40))

            except requests.exceptions.ConnectionError:
                print("Không thể kết nối tới Server. Đang thử lại...")
            except Exception as e:
                print(f"Error: {e}")

            time.sleep(0.2) # 5 FPS refresh rate

    except KeyboardInterrupt:
        print("\nĐã dừng theo dõi.")

if __name__ == "__main__":
    monitor()
