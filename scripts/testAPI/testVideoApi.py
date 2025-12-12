import requests
import time
import sys

# IP VPS của bạn
API_URL = "http://192.168.0.200:8000"
# Nếu chạy ngay trên VPS mà lỗi kết nối, hãy thử đổi thành "http://localhost:8000"

VIDEO_PATH = "./uploads/fight1.avi" 

def test_upload():
    print(f"1. Đang upload video: {VIDEO_PATH} ...")
    
    try:
        with open(VIDEO_PATH, "rb") as f:
            # Gửi POST request tới endpoint /detect
            response = requests.post(f"{API_URL}/detect_video", files={"file": f})
            
        if response.status_code != 200:
            print(f"Lỗi Upload: {response.text}")
            return

        data = response.json()
        task_id = data.get("task_id")
        print(f"Upload thành công! Task ID: {task_id}")
        
        # Polling để chờ kết quả từ Celery Worker
        print("2. Đang chờ Worker xử lý (Polling)...")
        while True:
            # Kiểm tra trạng thái task
            res = requests.get(f"{API_URL}/result/{task_id}")
            result = res.json()
            
            status = result.get("status")
            print(f"   Status: {status}")
            
            if status == "Success":
                print("\nKẾT QUẢ CUỐI CÙNG:")
                print(result["result"])
                break
            elif status == "Failed":
                print("\nXỬ LÝ THẤT BẠI:")
                print(result.get("error"))
                break
            
            # Chờ 1 giây rồi hỏi lại
            time.sleep(1)

    except FileNotFoundError:
        print("Không tìm thấy file video để test.")
    except Exception as e:
        print(f"Lỗi kết nối: {e}")

if __name__ == "__main__":
    test_upload()
