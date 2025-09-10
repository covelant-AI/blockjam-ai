import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def send_ntfy_notification(message: str):
    ntfy_topic = os.getenv("NTFY_TOPIC")

    if ntfy_topic:
        response = requests.post(
            f"https://ntfy.sh/{ntfy_topic}",
            data=message.encode('utf-8')
        )
        print("Push notification sent!" if response.status_code == 200 else "Failed to send push.")




if __name__ == "__main__":
    send_ntfy_notification("Hello, world!")