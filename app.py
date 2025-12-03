import os
import cv2
import numpy as np
import subprocess
import platform
import threading
import time
import uuid
import shutil
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
from color.deoldify import DEOLDIFY
import onnxruntime as rt

rt.set_default_logger_severity(3)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'temp'
app.config['RESULT_FOLDER'] = 'temp'

# Create temp directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Available models
AVAILABLE_MODELS = {
    'artistic': {
        'name': 'Artistic',
        'path': 'color/ColorizeArtistic_dyn.onnx',
        'description': 'More vibrant colors, less stable',
        'for': ['image', 'video']
    },
    'stable': {
        'name': 'Stable',
        'path': 'color/ColorizeStable_dyn.onnx',
        'description': 'More stable colors, less vibrant',
        'for': ['image', 'video']
    },
    'video': {
        'name': 'Video',
        'path': 'color/DeoldifyVideo_dyn.onnx',
        'description': 'Optimized for video processing',
        'for': ['video']
    }
}

# Model cache - store initialized models
model_cache = {}

def get_model(model_type, device='auto'):
    """Get or initialize a model"""
    logger.info(f"Getting model: {model_type}, device: {device}")
    
    if model_type not in AVAILABLE_MODELS:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_info = AVAILABLE_MODELS[model_type]
    model_path = model_info['path']
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Determine device
    if device == 'auto':
        # Check cache for both CUDA and CPU versions
        cache_key_cuda = f"{model_type}_cuda"
        cache_key_cpu = f"{model_type}_cpu"
        
        # Try CUDA cache first
        if cache_key_cuda in model_cache:
            logger.info(f"Using cached {model_type} model on CUDA")
            return model_cache[cache_key_cuda]
        
        # Try CPU cache
        if cache_key_cpu in model_cache:
            logger.info(f"Using cached {model_type} model on CPU")
            return model_cache[cache_key_cpu]
        
        # Try CUDA first, then CPU
        try:
            logger.info(f"Initializing {model_info['name']} model on CUDA...")
            colorizer = DEOLDIFY(model_path=model_path, device="cuda")
            logger.info(f"{model_info['name']} model initialized on CUDA")
            model_cache[cache_key_cuda] = colorizer
            return colorizer
        except Exception as e:
            logger.warning(f"CUDA failed for {model_type}, trying CPU: {e}")
            colorizer = DEOLDIFY(model_path=model_path, device="cpu")
            logger.info(f"{model_info['name']} model initialized on CPU")
            model_cache[cache_key_cpu] = colorizer
            return colorizer
    else:
        cache_key = f"{model_type}_{device}"
        if cache_key in model_cache:
            logger.info(f"Using cached {model_type} model on {device}")
            return model_cache[cache_key]
        
        logger.info(f"Initializing {model_info['name']} model on {device}...")
        colorizer = DEOLDIFY(model_path=model_path, device=device)
        logger.info(f"{model_info['name']} model initialized on {device}")
        model_cache[cache_key] = colorizer
        return colorizer

# Initialize default models on startup
default_image_colorizer = None
default_video_colorizer = None

try:
    logger.info("Initializing default image model (artistic)...")
    default_image_colorizer = get_model('artistic')
except Exception as e:
    logger.error(f"Error initializing default image model: {e}")

try:
    logger.info("Initializing default video model (stable)...")
    default_video_colorizer = get_model('stable')
except Exception as e:
    logger.error(f"Error initializing default video model: {e}")

# Progress tracking for video processing
video_progress = {}

def resize_image(image, max_width=1920, max_height=1080):
    """Resize image if it's too large (adapted from image_GUI.py)"""
    height, width = image.shape[:2]
    if height > max_height or width > max_width:
        scale = min(max_height/height, max_width/width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    return image

def adjust_saturation(image, saturation_factor):
    """Adjust saturation of image (from image_GUI.py)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * saturation_factor, 0, 255).astype(np.uint8)
    adjusted_hsv = cv2.merge([h, s, v])
    adjusted_bgr = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
    return adjusted_bgr

def process_image_file(file_path, render_factor=8, max_size=1920, model_type='artistic'):
    """Process image file (adapted from image_GUI.py process_image function)"""
    logger.info(f"Processing image: {file_path}, model: {model_type}, render_factor: {render_factor}")
    render_factor = render_factor * 32
    
    image = cv2.imread(file_path)
    if image is None:
        logger.error(f"Failed to load image from {file_path}")
        raise ValueError("Failed to load the image.")
    
    logger.info(f"Image loaded: {image.shape}")
    
    # Resize the image if it's too big
    original_shape = image.shape
    image = resize_image(image, max_width=max_size, max_height=max_size)
    if image.shape != original_shape:
        logger.info(f"Image resized from {original_shape} to {image.shape}")
    
    # Get the appropriate model
    logger.info(f"Getting model: {model_type}")
    colorizer = get_model(model_type)
    
    logger.info(f"Colorizing image with render_factor: {render_factor}")
    colorized = colorizer.colorize(image, render_factor)
    logger.info("Image colorization completed")
    
    # Convert the OpenCV BGR image to RGB
    colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    
    return colorized_rgb

def process_video_file(source_path, result_path, render_factor=8, keep_audio=False, task_id=None, model_type='stable'):
    """Process video file (adapted from video_GUI.py colorize_video function)"""
    logger.info(f"Processing video: {source_path}, model: {model_type}, render_factor: {render_factor}, keep_audio: {keep_audio}, task_id: {task_id}")
    render_factor = render_factor * 32
    
    video = cv2.VideoCapture(source_path)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Video properties: {w}x{h}, {n_frames} frames, {fps} fps")
    
    # Get the appropriate model
    logger.info(f"Getting model: {model_type}")
    colorizer = get_model(model_type)
    
    # Create preview directory for this task
    preview_dir = None
    if task_id:
        video_progress[task_id] = {'current': 0, 'total': n_frames, 'status': 'processing', 'previews': []}
        preview_dir = os.path.join(app.config['RESULT_FOLDER'], f'previews_{task_id}')
        os.makedirs(preview_dir, exist_ok=True)
    
    if keep_audio:
        temp_video_path = os.path.join(app.config['RESULT_FOLDER'], f'temp_{task_id}.mp4')
        writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))
    else:
        writer = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))
    
    preview_interval = 1  # Generate preview every frame
    
    logger.info(f"Starting video processing: {n_frames} frames to process")
    start_time = time.time()
    
    for frame_idx in range(n_frames):
        ret, frame = video.read()
        if not ret:
            logger.warning(f"Failed to read frame {frame_idx}")
            break
        
        if frame_idx == 0:
            logger.info(f"Processing first frame, shape: {frame.shape}")
        
        result = colorizer.colorize(frame, render_factor)
        writer.write(result)
        
        # Generate preview frame every N frames
        if task_id and preview_dir and (frame_idx % preview_interval == 0 or frame_idx == n_frames - 1):
            # Resize preview to reasonable size (max 400px width)
            preview_width = min(400, w)
            preview_height = int(h * (preview_width / w))
            preview_frame = cv2.resize(result, (preview_width, preview_height))
            
            # Save preview as JPEG
            preview_filename = f'preview_{frame_idx:06d}.jpg'
            preview_path = os.path.join(preview_dir, preview_filename)
            cv2.imwrite(preview_path, preview_frame)
            
            # Update progress with preview info
            if 'previews' not in video_progress[task_id]:
                video_progress[task_id]['previews'] = []
            video_progress[task_id]['previews'].append({
                'frame': frame_idx,
                'filename': preview_filename
            })
        
        if task_id:
            video_progress[task_id]['current'] = frame_idx + 1
        
        # Log progress every 10% or every 100 frames
        if (frame_idx + 1) % max(1, n_frames // 10) == 0 or (frame_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            fps_processing = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"Progress: {frame_idx + 1}/{n_frames} frames ({100*(frame_idx+1)//n_frames}%), {fps_processing:.2f} fps")
    
    writer.release()
    video.release()
    
    elapsed_total = time.time() - start_time
    logger.info(f"Video processing completed: {n_frames} frames in {elapsed_total:.2f}s ({n_frames/elapsed_total:.2f} fps)")
    
    if keep_audio:
        # Lossless remuxing audio/video (from video_GUI.py)
        # Try local tools directory first, then system PATH
        if platform.system() == 'Windows':
            local_ffmpeg = os.path.join('tools', 'ffmpeg.exe')
            if os.path.exists(local_ffmpeg):
                ffmpeg_exe = local_ffmpeg
            else:
                ffmpeg_exe = 'ffmpeg.exe'
        else:
            ffmpeg_exe = 'ffmpeg'
        
        # Check if source video has audio stream using ffprobe
        has_audio = False
        try:
            # Use ffprobe if available, otherwise use ffmpeg
            if platform.system() == 'Windows':
                local_ffprobe = os.path.join('tools', 'ffprobe.exe')
                probe_exe = local_ffprobe if os.path.exists(local_ffprobe) else 'ffprobe.exe'
            else:
                probe_exe = 'ffprobe'
            
            probe_cmd = [probe_exe, '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', source_path]
            probe_process = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
            has_audio = probe_process.returncode == 0 and 'audio' in probe_process.stdout.lower()
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            # If probe fails or ffprobe not found, try with ffmpeg
            try:
                probe_cmd = [ffmpeg_exe, '-i', source_path, '-hide_banner', '-f', 'null', '-']
                probe_process = subprocess.run(probe_cmd, capture_output=True, text=True, stderr=subprocess.STDOUT, timeout=5)
                has_audio = 'Audio:' in probe_process.stdout
            except Exception:
                # If all fails, assume no audio and proceed without it
                has_audio = False
        
        try:
            if has_audio:
                # Video has audio - remux with audio
                ffmpeg_cmd = [
                    ffmpeg_exe, '-y',
                    '-i', source_path,
                    '-i', temp_video_path,
                    '-c:v', 'copy',
                    '-c:a', 'libmp3lame', '-ac', '2', '-ar', '44100', '-ab', '128k',
                    '-map', '1:v:0',  # Video from second input (colorized video)
                    '-map', '0:a:0?',  # Audio from first input (original), ? makes it optional
                    '-shortest',
                    result_path
                ]
            else:
                # Video has no audio - just copy the colorized video
                ffmpeg_cmd = [
                    ffmpeg_exe, '-y',
                    '-i', temp_video_path,
                    '-c:v', 'copy',
                    '-an',  # No audio
                    result_path
                ]
            
            # Use subprocess.run with list instead of shell command for better error handling
            process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
            if process.returncode != 0:
                error_msg = process.stderr if process.stderr else f"ffmpeg failed with exit code {process.returncode}"
                # If audio mapping failed, try without audio
                if has_audio and ('0:a:0?' in str(ffmpeg_cmd) or 'map' in str(ffmpeg_cmd)):
                    ffmpeg_cmd_no_audio = [
                        ffmpeg_exe, '-y',
                        '-i', temp_video_path,
                        '-c:v', 'copy',
                        '-an',
                        result_path
                    ]
                    process_no_audio = subprocess.run(ffmpeg_cmd_no_audio, capture_output=True, text=True, check=False)
                    if process_no_audio.returncode != 0:
                        raise Exception(f"ffmpeg failed: {process_no_audio.stderr if process_no_audio.stderr else f'exit code {process_no_audio.returncode}'}")
                else:
                    raise Exception(f"ffmpeg failed: {error_msg}")
        except FileNotFoundError:
            raise Exception("ffmpeg not found. Please install ffmpeg in the tools/ directory or ensure it's in your PATH, or disable audio option.")
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
    
    if task_id:
        video_progress[task_id]['status'] = 'completed'

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    models_info = {}
    for key, model in AVAILABLE_MODELS.items():
        models_info[key] = {
            'name': model['name'],
            'description': model['description'],
            'for': model['for']
        }
    return jsonify(models_info)

@app.route('/api/colorize/image', methods=['POST'])
def colorize_image():
    """Process uploaded image"""
    try:
        logger.info("=== Image colorization request received ===")
        
        if 'file' not in request.files:
            logger.error("No file provided in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        render_factor = int(request.form.get('render_factor', 8))
        model_type = request.form.get('model_type', 'artistic')
        
        logger.info(f"Request parameters: filename={file.filename}, render_factor={render_factor}, model_type={model_type}")
        
        # Validate model type - only artistic and stable for images
        if model_type not in ['artistic', 'stable']:
            logger.warning(f"Invalid model type {model_type}, using 'artistic'")
            model_type = 'artistic'
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'upload_{uuid.uuid4()}_{filename}')
        logger.info(f"Saving uploaded file to: {filepath}")
        file.save(filepath)
        
        try:
            # Process image
            start_time = time.time()
            colorized_rgb = process_image_file(filepath, render_factor=render_factor, model_type=model_type)
            processing_time = time.time() - start_time
            logger.info(f"Image processing completed in {processing_time:.2f}s")
            
            # Save result
            result_filename = f'colorized_{uuid.uuid4()}_{os.path.splitext(filename)[0]}.jpg'
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            logger.info(f"Saving result to: {result_path}")
            cv2.imwrite(result_path, cv2.cvtColor(colorized_rgb, cv2.COLOR_RGB2BGR))
            
            logger.info(f"Image colorization successful: {result_filename}")
            return jsonify({
                'success': True,
                'filename': result_filename,
                'message': 'Image colorized successfully'
            })
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                logger.info(f"Cleaning up uploaded file: {filepath}")
                os.remove(filepath)
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        logger.error(f"Error processing image: {error_msg}")
        logger.error(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/api/colorize/video', methods=['POST'])
def colorize_video():
    """Process uploaded video"""
    try:
        logger.info("=== Video colorization request received ===")
        
        if 'file' not in request.files:
            logger.error("No file provided in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        render_factor = int(request.form.get('render_factor', 8))
        keep_audio = request.form.get('keep_audio', 'false').lower() == 'true'
        model_type = request.form.get('model_type', 'stable')
        
        logger.info(f"Request parameters: filename={file.filename}, render_factor={render_factor}, keep_audio={keep_audio}, model_type={model_type}")
        
        # Validate model type (only stable and video for videos)
        if model_type not in ['stable', 'video']:
            logger.warning(f"Invalid model type {model_type}, using 'stable'")
            model_type = 'stable'
        
        # Generate task ID for progress tracking
        task_id = str(uuid.uuid4())
        logger.info(f"Task ID: {task_id}")
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'upload_{task_id}_{filename}')
        logger.info(f"Saving uploaded file to: {filepath}")
        file.save(filepath)
        
        # Save result path
        result_filename = f'colorized_{task_id}_{os.path.splitext(filename)[0]}.mp4'
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        
        # Store result filename in progress for later retrieval
        if task_id not in video_progress:
            video_progress[task_id] = {}
        video_progress[task_id]['result_filename'] = result_filename
        
        # Process video in background thread
        def process():
            try:
                process_video_file(filepath, result_path, render_factor=render_factor, 
                                 keep_audio=keep_audio, task_id=task_id, model_type=model_type)
                # Clean up uploaded file after processing
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                if task_id in video_progress:
                    video_progress[task_id]['status'] = 'error'
                    video_progress[task_id]['error'] = str(e)
        
        thread = threading.Thread(target=process)
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Video processing started'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    """Get video processing progress"""
    if task_id not in video_progress:
        return jsonify({'error': 'Task not found'}), 404
    
    progress = video_progress[task_id]
    response = {
        'current': progress['current'],
        'total': progress['total'],
        'status': progress['status']
    }
    
    # Include preview frames if available
    if 'previews' in progress:
        response['previews'] = progress['previews']
    
    if progress['status'] == 'completed':
        # Get result filename from stored value or search for it
        if 'result_filename' in progress:
            result_filename = progress['result_filename']
            filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            if os.path.exists(filepath):
                response['filename'] = result_filename
            else:
                # File not found, try to locate it
                result_files = [f for f in os.listdir(app.config['RESULT_FOLDER']) if task_id in f and (f.endswith('.mp4') or f.endswith('.avi') or f.endswith('.mkv'))]
                if result_files:
                    response['filename'] = result_files[0]
                else:
                    response['error'] = f'Result file not found: {result_filename}'
                    response['status'] = 'error'
        else:
            # Fallback: search for files containing the task_id
            result_files = [f for f in os.listdir(app.config['RESULT_FOLDER']) if task_id in f and (f.endswith('.mp4') or f.endswith('.avi') or f.endswith('.mkv'))]
            if result_files:
                response['filename'] = result_files[0]
            else:
                # Try to find any video file that might be the result
                all_files = [f for f in os.listdir(app.config['RESULT_FOLDER']) if f.endswith(('.mp4', '.avi', '.mkv'))]
                if all_files:
                    # Use the most recent file
                    all_files.sort(key=lambda f: os.path.getmtime(os.path.join(app.config['RESULT_FOLDER'], f)), reverse=True)
                    response['filename'] = all_files[0]
                else:
                    response['error'] = 'No result file found'
                    response['status'] = 'error'
        
        # Clean up preview directory after completion
        preview_dir = os.path.join(app.config['RESULT_FOLDER'], f'previews_{task_id}')
        if os.path.exists(preview_dir):
            try:
                shutil.rmtree(preview_dir)
            except Exception as e:
                print(f"Error cleaning up preview directory: {e}")
        
        response['progress'] = 100
    elif progress['status'] == 'error':
        response['error'] = progress.get('error', 'Unknown error')
        response['progress'] = 0
    else:
        response['progress'] = int((progress['current'] / progress['total']) * 100) if progress['total'] > 0 else 0
    
    return jsonify(response)

@app.route('/api/preview/<task_id>/<filename>', methods=['GET'])
def get_preview(task_id, filename):
    """Get preview frame image"""
    from urllib.parse import unquote
    filename = unquote(filename)
    preview_dir = os.path.join(app.config['RESULT_FOLDER'], f'previews_{task_id}')
    filepath = os.path.join(preview_dir, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Preview not found'}), 404
    
    return send_file(filepath, mimetype='image/jpeg')

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download processed file"""
    # Decode URL-encoded filename
    from urllib.parse import unquote
    filename = unquote(filename)
    
    filepath = os.path.join(app.config['RESULT_FOLDER'], filename)
    
    # Debug: log the file path
    print(f"Download request for: {filename}")
    print(f"Full path: {filepath}")
    print(f"File exists: {os.path.exists(filepath)}")
    
    if not os.path.exists(filepath):
        # List available files for debugging
        available_files = os.listdir(app.config['RESULT_FOLDER'])
        print(f"Available files in {app.config['RESULT_FOLDER']}: {available_files}")
        return jsonify({'error': f'File not found: {filename}'}), 404
    
    return send_file(filepath, as_attachment=True)

@app.route('/api/cleanup/<filename>', methods=['POST'])
def cleanup_file(filename):
    """Clean up processed file after download"""
    # Decode URL-encoded filename
    from urllib.parse import unquote
    filename = unquote(filename)
    
    filepath = os.path.join(app.config['RESULT_FOLDER'], filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"Cleaned up file: {filename}")
        except Exception as e:
            print(f"Error cleaning up file {filename}: {e}")
    return jsonify({'success': True})

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Starting Colorizer Web Application")
    logger.info("=" * 50)
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Result folder: {app.config['RESULT_FOLDER']}")
    logger.info(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f} MB")
    logger.info("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)

