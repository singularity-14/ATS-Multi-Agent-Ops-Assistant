import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("NVIDIA_API_KEY", "")

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "application/json"
}

payload = {
    "model": "mistralai/mistral-medium-3.5-128b",
    "reasoning_effort": "low",
    "messages": [{"role":"user","content":"Hello"}],
    "max_tokens": 100,
    "temperature": 0.70,
    "top_p": 1.00,
    "stream": False
}

response = requests.post(invoke_url, headers=headers, json=payload)
print(f"Status Code: {response.status_code}")
print(f"Response Body: {response.text}")
