# Colorizer Web App

Modern web application for colorizing black & white images and videos using DeOldify ONNX models.

## Features

- 🖼️ **Image Colorization** - Transform black & white photos into color
- 🎬 **Video Colorization** - Colorize entire videos with live preview
- 🎨 **3 Models** - Choose between Artistic, Stable, or Video-optimized models
- ⚡ **Real-time Preview** - See frames as they're processed
- 🎛️ **Adjustable Quality** - Control render factor for quality vs speed

## Installation

### 1. Prerequisites

- **Python 3.7+** - [Download Python](https://www.python.org/downloads/)
- **FFmpeg** - Required for video audio processing
  - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - Place `ffmpeg.exe` in the `tools/` directory (Windows)
  - Or ensure `ffmpeg` is in your system PATH

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `flask` - Web framework
- `opencv-python` - Image/video processing
- `numpy` - Numerical operations
- `onnxruntime` - ONNX model inference
- `onnx` - ONNX model support
- `tqdm` - Progress bars

### 3. Download ONNX Models

You need to download the ONNX model files and place them in the `color/` directory:

**Required Models:**
- `ColorizeArtistic_dyn.onnx` - For artistic image colorization
- `ColorizeStable_dyn.onnx` - For stable colorization (images & videos)
- `DeoldifyVideo_dyn.onnx` - Optimized for video processing

**Download links:**
- Models available at: [Google Drive](https://drive.google.com/drive/folders/1bU9Zj7zGVEujIzvDTb1b9cyWU3s__WQR?usp=sharing)

**Directory structure after setup:**
```
color/
├── ColorizeArtistic_dyn.onnx
├── ColorizeStable_dyn.onnx
├── DeoldifyVideo_dyn.onnx
└── deoldify.py
```

### 4. Setup FFmpeg (for video audio)

**Windows:**
- Download `ffmpeg.exe` from [ffmpeg.org](https://ffmpeg.org/download.html)
- Place it in the `tools/` directory
- The application will automatically use `tools/ffmpeg.exe`

**Linux/Mac:**
- Install via package manager: `sudo apt install ffmpeg` or `brew install ffmpeg`
- Ensure `ffmpeg` is in your system PATH

### 5. Verify Installation

Check that all required files are present:
```
color/
  ├── ColorizeArtistic_dyn.onnx ✓
  ├── ColorizeStable_dyn.onnx ✓
  ├── DeoldifyVideo_dyn.onnx ✓
  └── deoldify.py ✓

tools/
  └── ffmpeg.exe ✓ (Windows only, or in PATH)
```

## Quick Start

1. **Run the application:**
   ```bash
   python app.py
   ```

2. **Open in browser:**
   ```
   http://localhost:5000
   ```

3. **Upload and colorize:**
   - Drag & drop an image or video
   - Select model and adjust settings
   - Click "Colorize" and wait for processing
   - Download the result

## Project Structure

```
Colorizer/
├── app.py              # Flask web application
├── templates/          # Web interface (HTML/CSS/JS)
├── color/              # ONNX models and processing code
│   ├── *.onnx         # Model files (download separately)
│   └── deoldify.py    # Colorization processing code
├── tools/             # FFmpeg executables (Windows)
│   └── ffmpeg.exe     # Download and place here
├── temp/              # Temporary files (auto-created)
└── archive/           # Legacy GUI/CLI applications
```

## Usage

1. **Upload** - Drag & drop an image or video in the left preview area
2. **Configure** - Select model type and adjust render factor (1-10)
3. **Process** - Click "Colorize" button
4. **Preview** - Watch live preview (for videos) or see result (for images)
5. **Download** - Click "Download" button to save the colorized file

## Models

- **Artistic** (`ColorizeArtistic_dyn.onnx`)
  - More vibrant colors, less stable
  - Best for: Images
  - Works on: Images only

- **Stable** (`ColorizeStable_dyn.onnx`)
  - More stable colors, less vibrant
  - Best for: Both images and videos
  - Works on: Images and videos

- **Video** (`DeoldifyVideo_dyn.onnx`)
  - Optimized for video processing
  - Best for: Videos
  - Works on: Videos only

## System Requirements

### Minimum
- Python 3.7+
- 4GB RAM
- CPU processing (slower)

### Recommended
- Python 3.8+
- 8GB+ RAM
- NVIDIA GPU with CUDA support (much faster)
- SSD storage for faster I/O

## License

Based on [DeOldify](https://github.com/jantic/DeOldify)

