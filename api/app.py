import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, 'models', 'saved_models', 'colorize_epoch_5.pth')
sys.path.insert(0, project_root)

import threading
import traceback
from flask import Flask, render_template, request
import cv2
import numpy as np
import torch
from models.colorize_unet import UNetColorize

def delete_file_after_delay(file_path, delay=10):
    def remove_file():
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    timer = threading.Timer(delay, remove_file)
    timer.start()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Limit file upload size (e.g., 2 MB)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetColorize().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def colorize_image(gray_img_path, result_path):
    try:
        gray = cv2.imread(gray_img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("Could not read image file")
        gray = cv2.resize(gray, (256, 256)) / 255.0
        gray_tensor = torch.FloatTensor(gray[None, None, ...]).to(device)
        with torch.no_grad():
            pred_ab = model(gray_tensor).cpu().numpy()[0]
        pred_ab = (pred_ab + 1) * 128
        pred_ab = pred_ab.transpose(1, 2, 0).astype(np.uint8)
        L = (gray * 255).astype(np.uint8)
        LAB_pred = np.zeros((256, 256, 3), dtype=np.uint8)
        LAB_pred[:, :, 0] = L
        LAB_pred[:, :, 1:] = pred_ab
        colorized_bgr = cv2.cvtColor(LAB_pred, cv2.COLOR_LAB2BGR)
        cv2.imwrite(result_path, colorized_bgr)
    except Exception as e:
        print(f"Error in colorize_image: {e}")
        traceback.print_exc()
        raise

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            file = request.files['file']
            if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                result_path = os.path.join(app.config['RESULT_FOLDER'], 'colorized_' + file.filename)
                print("Saving to:", upload_path, result_path)
                file.save(upload_path)
                colorize_image(upload_path, result_path)

                delete_file_after_delay(upload_path, delay=10)
                delete_file_after_delay(result_path, delay=10)

                return render_template('index.html',
                                    upload=file.filename,
                                    result='colorized_' + file.filename)
        return render_template('index.html')
    except Exception as e:
        print("Exception occurred:", e)
        traceback.print_exc()
        return "An error occurred: {}".format(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))