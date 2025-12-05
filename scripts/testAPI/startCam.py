import requests

# IP của server
SERVER_URL = "http://103.78.3.29:8000/camera/start"

# Nhập số lượng camera muốn start
num_cameras = int(input("Nhập số lượng camera: "))

for i in range(1, num_cameras + 1):
    camera_id = f"cam{i:02d}"  # cam01, cam02,...
    rtsp_url = f"rtsp://103.78.3.29:8554/{camera_id}"
    
    try:
        response = requests.post(SERVER_URL, params={
            "camera_id": camera_id,
            "rtsp_url": rtsp_url
        })
        if response.status_code == 200:
            print(f"{camera_id} started successfully!")
        else:
            print(f"Failed to start {camera_id}: {response.text}")
    except Exception as e:
        print(f"Error starting {camera_id}: {e}")
