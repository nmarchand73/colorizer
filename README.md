# Colorizer Web App

Modern web application for colorizing black & white images and videos using DeOldify ONNX models.

## Features

- 🖼️ **Image Colorization** - Transform black & white photos into color
- 🎬 **Video Colorization** - Colorize entire videos with live preview
- 🎨 **3 Models** - Choose between Artistic, Stable, or Video-optimized models
- ⚡ **Real-time Preview** - See frames as they're processed
- 🎛️ **Adjustable Quality** - Control render factor for quality vs speed

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Open in browser:**
   ```
   http://localhost:5000
   ```

## Requirements

- Python 3.7+
- ONNX Runtime (CPU or CUDA)
- FFmpeg (for video audio processing)
- Models in `color/` directory:
  - `ColorizeArtistic_dyn.onnx`
  - `ColorizeStable_dyn.onnx`
  - `DeoldifyVideo_dyn.onnx`

## Project Structure

```
Colorizer/
├── app.py              # Flask web application
├── templates/          # Web interface
├── color/             # ONNX models and processing code
├── tools/              # FFmpeg executables
├── temp/               # Temporary files
└── archive/            # Legacy GUI/CLI applications
```

## Usage

1. Upload an image or video using drag & drop
2. Select model and adjust render factor
3. Click "Colorize" and wait for processing
4. Download the colorized result

## Models

- **Artistic** - More vibrant colors, best for images
- **Stable** - More stable colors, works for both images and videos
- **Video** - Optimized for video processing

## License

Based on [DeOldify](https://github.com/jantic/DeOldify)

