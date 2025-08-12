import runpod
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.playground.shot_type import test_shot_type

def handler(job):
    test_shot_type()
runpod.serverless.start({"handler":handler})