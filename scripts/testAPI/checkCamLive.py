import requests

resp = requests.get("http://103.78.3.32:8000/system/active_cameras")
print(resp.json())
