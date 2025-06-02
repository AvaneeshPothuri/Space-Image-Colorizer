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

model_path = os.path.join(project_root, 'models', 'saved_models', 'best_model.pth')
sys.path.insert(0, project_root)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    return f"{process.memory_info().rss / 1024 / 1024:.2f} MB"

def delete_file_after_delay(file_path, delay=60):
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

# Enhanced CORS configuration
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "https://space-image-colorizer.netlify.app",
                "http://localhost:5173"
            ],
            "methods": ["POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "supports_credentials": True
        }
    }
)

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    allowed_origins = [
        "https://space-image-colorizer.netlify.app",
        "http://localhost:5173"
    ]
    origin = request.headers.get('Origin', '')
    
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Model loading
try:
    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetColorize().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded on {device}")
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    traceback.print_exc()
    raise

def colorize_image(gray_img_path, result_path):
    try:
        print("Starting colorization...")
        start_time = time.time()
        
        # Image processing logic
        gray = cv2.imread(gray_img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("Invalid image file")
        
        gray = cv2.resize(gray, (256, 256)) / 255.0
        
        with torch.no_grad():
            gray_tensor = torch.FloatTensor(gray[None, None, ...]).to(device)
            pred_ab = model(gray_tensor).cpu().numpy().squeeze()
            del gray_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Post-processing
        pred_ab = (pred_ab + 1) * 128
        pred_ab = pred_ab.transpose(1, 2, 0).astype(np.uint8)
        
        L = (gray * 255).astype(np.uint8)
        LAB_pred = np.zeros((256, 256, 3), dtype=np.uint8)
        LAB_pred[:, :, 0] = L
        LAB_pred[:, :, 1:] = pred_ab
        
        colorized_bgr = cv2.cvtColor(LAB_pred, cv2.COLOR_LAB2BGR)
        cv2.imwrite(result_path, colorized_bgr)

        print(f"Colorization completed in {time.time()-start_time:.2f}s")
        return True

    except Exception as e:
        print(f"Colorization error: {str(e)}")
        traceback.print_exc()
        raise

@app.route('/api/colorize', methods=['POST'])
def handle_colorization():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'Empty file'}), 400

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type'}), 400

        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        result_path = os.path.join(app.config['RESULT_FOLDER'], f'colorized_{file.filename}')
        
        file.save(upload_path)
        colorize_image(upload_path, result_path)
        
        delete_file_after_delay(upload_path)
        delete_file_after_delay(result_path)

        return jsonify({
            'upload': f'/static/uploads/{file.filename}',
            'result': f'/static/results/colorized_{file.filename}'
        })

    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/')
def health_check():
    return jsonify({'status': 'active', 'timestamp': time.time()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )
