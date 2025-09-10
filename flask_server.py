#!/usr/bin/env python3
"""
Simple Flask server for tennis analysis with single endpoint.
Accepts data in the same format as test_input.json
"""

import os
import sys
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from routers.analysis import handle_process_video
from ai_core.trackers.build_engine import build_engines
import ai_core

app = Flask(__name__)

# Global variable to store AI models (initialized once)
ai_models = None

def get_model_files(prefix):
    #all files in models directory with prefix
    all_files = [f for f in os.listdir("models") if f.startswith(prefix)]
    if len(all_files) > 1:
        #Choose the file which does not have the pt extension
        for file in all_files:
            if not file.endswith(".pt"):
                return "models/" + file
    elif len(all_files) == 1:
        return "models/" + all_files[0]
    else:
        raise Exception(f"No model files found for prefix {prefix}")

def initialize_models():
    """Initialize AI models once at startup"""
    global ai_models
    if ai_models is None:
        ai_models = ai_core.initialize_models(
            ball_tracker_model_path=get_model_files("best-balls"),
            player_tracker_model_path=get_model_files("best-player"),
            court_line_detector_model_path="models/model_tennis_court_det.pt",
            letr_court_line_detector_model_path="models/letr_best_checkpoint.pth",
        )
    return ai_models

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "tennis-analysis-simple"})

@app.route('/build-engines', methods=['POST'])
def build_engines_endpoint():
    """Build engines endpoint"""
    try:
        player_model_path, ball_model_path = build_engines("models/best-player.pt", "models/best-balls.pt")
        return jsonify({"status": "engines built", "player_model_path": player_model_path, "ball_model_path": ball_model_path})
    except Exception as e:
        return jsonify({"status": "engines build failed", "error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """
    Single endpoint that accepts data in test_input.json format
    Expected JSON format:
    {
        "input": {
            "route": "analysis/process_video",
            "data": {
                "video_url": "https://example.com/video.mp4",
                "video_id": "1",
                "features": ["DeadTimeDetection", "MatchSectioning"]
            }
        }
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate that the data has the expected structure
        if 'input' not in data:
            return jsonify({"error": "Missing 'input' field in request data"}), 400
        
        if 'data' not in data['input']:
            return jsonify({"error": "Missing 'data' field in input"}), 400
        
        # Initialize models if not already done
        models = initialize_models()
        
        # Process the video using the existing analysis handler
        result = handle_process_video(data, models)
        
        return jsonify({
            "success": True,
            "message": "Video analysis completed successfully",
            "result": result
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Video analysis failed: {str(e)}"
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        "service": "Tennis Analysis Simple Server",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Analyze tennis video (accepts test_input.json format)",
            "POST /build-engines": "Build TensorRT engines for player and ball models",
            "GET /health": "Health check",
            "GET /": "This information"
        },
        "example_request": {
            "input": {
                "route": "analysis/process_video",
                "data": {
                    "video_url": "https://example.com/video.mp4",
                    "video_id": "1",
                    "features": ["DeadTimeDetection", "MatchSectioning"]
                }
            }
        }
    })

if __name__ == '__main__':
    # Initialize models on startup
    print("Initializing AI models...")
    # initialize_models()
    print("AI models initialized successfully!")
    
    # Get configuration from environment variables
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    print(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
