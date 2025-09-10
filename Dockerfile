FROM python:3.11.1-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (models will be mounted at runtime)
COPY . .

#Remove models directory
RUN rm -rf models

# Make directories
RUN mkdir -p tracker_stubs input_videos output_videos output_images

# Command to run when the container starts
CMD ["python", "-u", "/handler.py"]