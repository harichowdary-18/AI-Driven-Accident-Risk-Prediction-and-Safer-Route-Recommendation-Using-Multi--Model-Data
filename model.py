import requests
ph="9080858902"
base_url = "https://loc-app-sathya-default-rtdb.asia-southeast1.firebasedatabase.app/hazards"
url = f"{base_url}/{ph}.json"
val=2
response = requests.put(url, json=val)

if response.status_code == 200:
    print("Value Updated")
    
else:
    print("Not Updated")