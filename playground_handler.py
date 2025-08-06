import runpod
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.playground.download import test_download_file_from_firebase

from dotenv import load_dotenv

def handler(job):
    test_download_file_from_firebase()
runpod.serverless.start({"handler":handler})