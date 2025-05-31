import sys
import os

# Set up paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, 'models', 'saved_models', 'colorize_epoch_5.pth')
sys.path.insert(0, project_root)

import threading
import traceback
import time
import psutil
from flask import Flask, render_template, request
from flask_cors import CORS
import cv2
import numpy as np
import torch
from models.colorize_unet import UNetColorize

def log_memory_usage():
    process = psutil.Process(os.getpid())
    return "{:.2f} MB".format(process.memory_info().rss / 1024 / 1024)

def delete_file_after_delay(file_path, delay=10):
    def remove_file():
        try:
            os.remove(file_path)
            print("Deleted:", file_path)
        except Exception as e:
            print("Error deleting", file_path, ":", str(e))
    timer = threading.Timer(delay, remove_file)
    timer.start()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Model loading with verification
try:
    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model = UNetColorize().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print("Failed to load model:", str(e))
    traceback.print_exc()
    raise

def colorize_image(gray_img_path, result_path):
    try:
        print("Starting colorization process")
        start_time = time.time()
        
        # Read and verify image
        print("Reading image from:", gray_img_path)
        gray = cv2.imread(gray_img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("Could not read image file")
            
        # Preprocess
        print("Original size:", gray.shape, "| Memory:", log_memory_usage())
        gray = cv2.resize(gray, (256, 256)) / 255.0
        
        # Convert to tensor
        print("Converting to tensor...")
        gray_tensor = torch.FloatTensor(gray[None, None, ...]).to(device)
        
        # Model inference
        print("Running model inference...")
        with torch.no_grad():
            pred_ab = model(gray_tensor).cpu().numpy()[0]
            
        # Post-process
        print("Post-processing...")
        pred_ab = (pred_ab + 1) * 128
        pred_ab = pred_ab.transpose(1, 2, 0).astype(np.uint8)
        
        # Create LAB image
        L = (gray * 255).astype(np.uint8)
        LAB_pred = np.zeros((256, 256, 3), dtype=np.uint8)
        LAB_pred[:, :, 0] = L
        LAB_pred[:, :, 1:] = pred_ab
        
        # Convert and save
        print("Saving result...")
        colorized_bgr = cv2.cvtColor(LAB_pred, cv2.COLOR_LAB2BGR)
        cv2.imwrite(result_path, colorized_bgr)
        
        print("Colorization completed in {:.2f}s | Memory: {}".format(time.time()-start_time, log_memory_usage()))
        
    except Exception as e:
        print("Colorization failed:", str(e))
        traceback.print_exc()
        raise

@app.route('/', methods=['GET', 'POST'])
def handle_requests():
    try:
        print("\nReceived", request.method, "request")
        print("Headers:", dict(request.headers))
        print("Form data:", dict(request.form))
        
        if request.method == 'POST':
            if 'file' not in request.files:
                return "No file uploaded", 400
                
            file = request.files['file']
            if not file or file.filename == '':
                return "Empty file submitted", 400

            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                return "Invalid file type", 400

            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            result_path = os.path.join(app.config['RESULT_FOLDER'], 'colorized_' + file.filename)
            
            print("Saving upload to:", upload_path)
            file.save(upload_path)
            print("File saved. Size:", os.path.getsize(upload_path), "bytes")
            
            print("Starting colorization...")
            colorize_image(upload_path, result_path)
            print("Colorization completed successfully")
            
            # Schedule cleanup
            delete_file_after_delay(upload_path, 10)
            delete_file_after_delay(result_path, 10)
            
            return render_template('index.html',
                                upload=file.filename,
                                result='colorized_' + file.filename)
            
        return render_template('index.html')
        
    except Exception as e:
        print("Critical error:", str(e))
        traceback.print_exc()
        return "Server Error: {}".format(str(e)), 500

if __name__ == '__main__':
    print("Starting server...")
    print("Working directory:", os.getcwd())
    print("Device:", device)
    print("Model architecture:", model)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))