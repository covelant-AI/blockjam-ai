# Covelant AI - Tennis Video Analysis System

A comprehensive AI-powered tennis video analysis system that provides real-time ball tracking, player detection, court keypoint detection, and match analytics. The system is designed for deployment on Theta Edge Cloud and uses state-of-the-art computer vision models for accurate tennis match analysis.

## 🎾 Features

- **Ball Detection & Tracking**: High-resolution YOLO model for precise ball detection with custom Botsort tracking
- **Player Detection & Tracking**: Advanced player detection and tracking across the court
- **Court Keypoint Detection**: LETR (Line Segment Detection Transformer) for accurate court line detection
- **Match Analytics**: Dead time detection, match sectioning, and performance metrics
- **Real-time Processing**: Optimized for edge computing deployment
- **RESTful API**: Simple Flask-based API for easy integration

## 🏗️ Architecture

### Core Components

1. **AI Core (`src/ai_core/`)**
   - **Ball & Player Tracker**: YOLO-based detection with custom Botsort tracking
   - **Court Line Detector**: Traditional computer vision approach
   - **LETR Court Detector**: Transformer-based line segment detection
   - **Speed Calculator**: Real-time speed calculations using homography

2. **Analysis Services (`src/services/analysis/`)**
   - **Core Analysis**: Main video processing pipeline
   - **Court Keypoints**: Court line detection and keypoint extraction
   - **Tracking**: Ball and player tracking implementation
   - **Sectioning**: Match sectioning based on ball detections

3. **Flask Server (`flask_server.py`)**
   - RESTful API endpoints
   - Model initialization and management
   - Request handling and response formatting

### AI Models Used

- **YOLO (You Only Look Once)**: Finetuned high-resolution model for ball and player detection
- **Botsort**: Custom tracking layer for robust object tracking across frames
- **LETR (Line Segment Detection Transformer)**: Deep learning model for court keypoint detection

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (required for AI models)
- Docker (for containerized deployment)

### Local Development Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Download AI models**
   Place the following model files in the `models/` directory:
   - `best-balls.pt` - Ball detection model
   - `best-player.pt` - Player detection model
   - `model_tennis_court_det.pt` - Court line detection model
   - `letr_best_checkpoint.pth` - LETR court detector model

4. **Run the Flask server**
   ```bash
   python flask_server.py
   ```

The server will start on `http://localhost:5000`

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -f Dockerfile.flask -t blockjam-ai .
   ```

2. **Run the container**
   ```bash
   docker run -p 5000:5000 \
     -e HOST=0.0.0.0 \
     -e PORT=5000 \
     -e DEBUG=False \
     blockjam-ai
   ```

## 📡 API Documentation

### Endpoints

#### `GET /`
Returns API information and available endpoints.

**Response:**
```json
{
  "service": "Tennis Analysis Simple Server",
  "version": "1.0.0",
  "endpoints": {
    "POST /analyze": "Analyze tennis video",
    "POST /build-engines": "Build TensorRT engines",
    "GET /health": "Health check",
    "GET /": "API information"
  }
}
```

#### `POST /analyze`
Analyzes a tennis video and returns comprehensive analytics.

**Request Body:**
```json
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
```

**Response:**
```json
{
  "success": true,
  "message": "Video analysis completed successfully",
  "result": {
    "timing_results": {
      "court_keypoints": 2.5,
      "tracking": 15.3
    }
  }
}
```

#### `POST /build-engines`
Builds TensorRT engines for optimized inference.

**Response:**
```json
{
  "status": "engines built",
  "player_model_path": "path/to/player.engine",
  "ball_model_path": "path/to/ball.engine"
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "tennis-analysis-simple"
}
```

## 🔧 Configuration

### Environment Variables

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 5000)
- `DEBUG`: Debug mode (default: False)
- `BACKEND_URL`: Backend service URL for data persistence
- `PYTHONPATH`: Python path for imports (set to `/app/src` in Docker)

### Model Configuration

The system automatically detects model files in the `models/` directory:
- Ball detection: Files starting with `best-balls`
- Player detection: Files starting with `best-player`
- Court detection: `model_tennis_court_det.pt`
- LETR detector: `letr_best_checkpoint.pth`

## 🚀 Theta Edge Cloud Deployment

### Prerequisites for Theta Edge Cloud

1. **Theta Edge Cloud Account**: Sign up at [Theta Edge Cloud](https://www.thetavideo.ai/)
2. **Docker Image**: Build and push your Docker image to a registry
3. **GPU Support**: Ensure your deployment uses GPU-enabled nodes

### Deployment Steps

1. **Build and tag your image**
   ```bash
   docker build -f Dockerfile.flask -t your-registry/blockjam-ai:latest .
   docker push your-registry/blockjam-ai:latest
   ```

2. **Deploy to Theta Edge Cloud**
   - Use the Theta Edge Cloud dashboard or CLI
   - Configure GPU resources (recommended: 1 GPU with 8GB+ VRAM)
   - Set environment variables as needed
   - Configure health checks using the `/health` endpoint

3. **Monitor deployment**
   - Check logs for model initialization
   - Verify GPU availability and CUDA compatibility
   - Test with sample video analysis requests

### Performance Optimization

- **Model Loading**: Models are loaded once at startup for optimal performance
- **GPU Memory**: Ensure sufficient VRAM for all models (recommended: 8GB+)
- **Batch Processing**: Configure worker processes based on available resources
- **Caching**: Implement caching for frequently accessed data

## 📊 Analysis Pipeline

### 1. Video Input Processing
- Video URL validation and download
- Convert Video to standard (30fps 720p)
- Frame extraction and preprocessing

### 2. Court Detection
- LETR-based court line detection
- Keypoint extraction and validation
- Homography matrix calculation for court mapping

### 3. Object Detection & Tracking
- YOLO-based ball and player detection
- Botsort tracking for robust object following
- Speed calculation using homography transformation

### 4. Match Analytics
- Dead time detection based on ball movement
- Match sectioning (sets, games, points)
- Performance metrics calculation

### 5. Results Processing
- Data aggregation and formatting
- Backend integration for data persistence
- Response generation and delivery

## 🛠️ Development

### Project Structure

```
blockjam-ai/
├── src/
│   ├── ai_core/                 # AI model implementations
│   │   ├── trackers/           # Ball and player tracking
│   │   ├── court_line_detector/ # Court detection
│   │   └── letr_court_line_detector/ # LETR implementation
│   ├── services/
│   │   └── analysis/           # Analysis services
│   ├── routers/                # API route handlers
│   ├── schemas/                # Data schemas
│   └── utils/                  # Utility functions
├── models/                     # AI model files (not in repo)
├── flask_server.py            # Main Flask application
├── requirements.txt           # Python dependencies
├── Dockerfile.flask          # Docker configuration
└── README.md                 # This file
```

**Note**: This system requires GPU acceleration for optimal performance. Ensure your deployment environment has CUDA-compatible GPUs available.
