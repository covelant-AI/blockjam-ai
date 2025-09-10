import os
import requests
import cv2
import tempfile
from datetime import datetime
import subprocess
import json
import shutil
from tqdm import tqdm
import re

def download_file_from_firebase(url, output_path):
    """
    Download a video from URL to the specified output path and convert to 720p 30fps.
    """
    try:
        # Download the video to a temporary file first with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        temp_path = tempfile.mktemp(prefix=f'tempfile_{timestamp}_', suffix='.mp4')
        
        # Download with progress bar
        with open(temp_path, 'wb') as f:
            with tqdm(total=total_size, desc="Downloading video", unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Convert video to 720p and 30fps
        convert_video_to_standard(temp_path, output_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return output_path
    except Exception as e:
        raise Exception(f"Failed to download and convert video: {str(e)}")

def convert_video_to_standard(input_path, output_path, target_resolution=(1280, 720), target_fps=30):
    """
    Convert video to target resolution and fps using FFmpeg with progress bar.
    """
    
    # Get original video properties using FFmpeg
    probe_cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json', 
        '-show_format', '-show_streams', input_path
    ]
    
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            raise Exception("No video stream found")
        
        original_width = int(video_stream['width'])
        original_height = int(video_stream['height'])
        original_fps = eval(video_stream['r_frame_rate'])  # e.g., "30/1" -> 30.0
        
        target_width, target_height = target_resolution
        
        needs_resolution_conversion = (original_width != target_width or 
                           original_height != target_height)
        
        needs_fps_conversion = (abs(original_fps - target_fps) > 0.1)
        
        if not needs_resolution_conversion and not needs_fps_conversion:
            # If no conversion needed, just copy the file
            shutil.copy2(input_path, output_path)
            return
        
        # Check if NVENC is available
        nvenc_available = False
        try:
            check_nvenc = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, check=True)
            if 'h264_nvenc' in check_nvenc.stdout:
                nvenc_available = True
        except:
            pass
        
        # Get video duration for progress calculation
        duration_cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', input_path
        ]
        
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
        duration = float(duration_result.stdout.strip())
        
        # Build FFmpeg command with progress output
        ffmpeg_cmd = [
            'ffmpeg', '-i', input_path, '-y',  # -y overwrites output
            '-progress', 'pipe:1',  # Output progress to stdout
            '-loglevel', 'error'    # Only show errors
        ]
        
        # Set output resolution
        ffmpeg_cmd.extend(['-vf', f'scale={target_width}:{target_height}'])
        
        # Set output fps
        ffmpeg_cmd.extend(['-r', str(target_fps)])
        
        # Set output codec and quality - use NVENC if available, otherwise CPU
        if nvenc_available:
            ffmpeg_cmd.extend(['-c:v', 'h264_nvenc', '-preset', 'fast'])
        else:
            ffmpeg_cmd.extend(['-c:v', 'libx264', '-preset', 'medium', '-crf', '23'])
        
        # Set output format
        ffmpeg_cmd.extend(['-c:a', 'copy'])  # Copy audio if present
        
        # Output file
        ffmpeg_cmd.append(output_path)
        
        # Run FFmpeg with progress tracking
        process = subprocess.Popen(
            ffmpeg_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Initialize progress bar
        with tqdm(total=100, desc="Converting video", unit="%") as pbar:
            last_progress = 0
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Parse progress from FFmpeg output
                    if 'out_time_ms=' in output:
                        time_match = re.search(r'out_time_ms=(\d+)', output)
                        if time_match:
                            current_time_ms = int(time_match.group(1))
                            current_time = current_time_ms / 1000000  # Convert to seconds
                            progress = min(100, (current_time / duration) * 100)
                            
                            # Update progress bar
                            progress_diff = progress - last_progress
                            if progress_diff > 0:
                                pbar.update(progress_diff)
                                last_progress = progress
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode != 0:
                stderr_output = process.stderr.read()
                raise Exception(f"FFmpeg conversion failed: {stderr_output}")
        
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg conversion failed: {e.stderr}")
    except Exception as e:
        raise Exception(f"Video conversion failed: {str(e)}")
    
