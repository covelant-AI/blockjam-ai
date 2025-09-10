import runpod
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.playground.compare_ball_trackers import compare_ball_trackers

def handler(job):
    compare_ball_trackers()
runpod.serverless.start({"handler":handler})