import requests
API_URL = "http://103.78.3.29:8000"
for i in range(1, 17):
    cam_id = f"cam{i:02d}"
    requests.post(f"{API_URL}/camera/stop", params={"camera_id": cam_id})
    print(f"Stopped {cam_id}")
