import sys
import os

current_file_path = os.path.abspath(__file__)      
api_directory = os.path.dirname(current_file_path)     
project_root = os.path.dirname(api_directory)
sys.path.insert(0, project_root)

import threading
import traceback
import time
import psutil
from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import torch
from models.colorize_unet import UNetColorize

model_path = os.path.join(project_root, 'models', 'saved_models', 'colorize_epoch_5.pth')
sys.path.insert(0, project_root)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    return f"{process.memory_info().rss / 1024 / 1024:.2f} MB"

def delete_file_after_delay(file_path, delay=60):  # Increased delay to 60 seconds
    def remove_file():
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")
    timer = threading.Timer(delay, remove_file)
    timer.start()

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB limit

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Model loading with verification
try:
    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Available devices: CPU{' + CUDA' if torch.cuda.is_available() else ''}")
    print(f"Using device: {device}")

    model = UNetColorize().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    traceback.print_exc()
    raise

def colorize_image(gray_img_path, result_path):
    try:
        print("Starting colorization process")
        start_time = time.time()
        print(f"Initial memory: {log_memory_usage()}")

        # Read and verify image
        print(f"Reading image from: {gray_img_path}")
        gray = cv2.imread(gray_img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("Could not read image file")

        # Resize early to reduce memory
        original_shape = gray.shape
        print(f"Original size: {original_shape}")
        gray = cv2.resize(gray, (256, 256)) / 255.0

        # Convert to tensor and process
        print("Converting to tensor...")
        with torch.no_grad():
            gray_tensor = torch.FloatTensor(gray[None, None, ...]).to(device)
            pred_ab = model(gray_tensor).cpu().numpy().squeeze()
            del gray_tensor  # Explicitly delete tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Post-process
        print("Post-processing...")
        pred_ab = (pred_ab + 1) * 128
        pred_ab = pred_ab.transpose(1, 2, 0).astype(np.uint8)

        # Create LAB image
        L = (gray * 255).astype(np.uint8)
        del gray  # Free memory

        LAB_pred = np.zeros((256, 256, 3), dtype=np.uint8)
        LAB_pred[:, :, 0] = L
        LAB_pred[:, :, 1:] = pred_ab

        # Convert and save
        print("Saving result...")
        colorized_bgr = cv2.cvtColor(LAB_pred, cv2.COLOR_LAB2BGR)
        cv2.imwrite(result_path, colorized_bgr)

        # Force garbage collection
        import gc
        gc.collect()

        print(f"Colorization completed in {time.time()-start_time:.2f}s")
        print(f"Final memory: {log_memory_usage()}")

    except Exception as e:
        print(f"Colorization failed: {str(e)}")
        traceback.print_exc()
        raise

@app.route('/api/colorize', methods=['POST'])
def handle_colorization():
    try:
        print("Received API request")
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'Empty file submitted'}), 400

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type'}), 400

        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'colorized_' + file.filename)

        print(f"Saving upload to: {upload_path}")
        file.save(upload_path)
        print(f"File size: {os.path.getsize(upload_path)} bytes")

        print("Starting colorization...")
        colorize_image(upload_path, result_path)
        print("Colorization completed successfully")

        # Schedule cleanup
        delete_file_after_delay(upload_path, 60)
        delete_file_after_delay(result_path, 60)

        return jsonify({
            'upload': f'/static/uploads/{file.filename}',
            'result': f'/static/results/colorized_{file.filename}'
        })

    except Exception as e:
        print(f"Critical error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/')
def health_check():
    return jsonify({'status': 'healthy', 'service': 'Space Image Colorizer'})

if __name__ == '__main__':
    print("Starting server...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Available devices: CPU{' + CUDA' if torch.cuda.is_available() else ''}")
    print(f"Model architecture:\n{model}")

    app.run(
        host='0.0.0.0', 
        port=8080, 
        debug=True, 
        use_reloader=False,
        threaded=True
    )