# Space Image Colorizer

## Project Overview

This project is a deep learning-based image colorization tool specifically designed for space images. It takes grayscale images of space (such as galaxies, nebulae, and stars) and colorizes them using a U-Net convolutional neural network architecture.

---

## Dataset

- **Total images:** 529
- **Split:**  
  - **Training images:** 423
  - **Validation images:** 106
- **Image resolution:** 256x256 pixels (resized and converted to grayscale for input)
- **Ground truth:** Original color images used for training and validation

- **Source:**  
  - Images are sourced from the [NASA Image and Video Library](https://images.nasa.gov/), which provides public domain space imagery.

---

## Model Architecture

- **Architecture:** U-Net style convolutional neural network
- **Input:** Grayscale image (L channel in LAB color space)
- **Output:** Predicted a and b color channels (LAB color space)
- **Purpose:** The model predicts color information from grayscale input, producing a colorized output.

---

## Training Details

- **Training epochs:** 10
- **Batch size:** 4
- **Loss function:** L1 loss (mean absolute error)
- **Optimizer:** Adam with learning rate 0.001
- **Training device:** CPU

---

## Model Evaluation Results

- **Peak Signal-to-Noise Ratio (PSNR):**
  - **Training PSNR:** 25.40 dB
  - **Validation PSNR:** 26.26 dB
- **Structural Similarity Index (SSIM):**
  - **Training SSIM:** 0.8230
  - **Validation SSIM:** 0.8300

These metrics indicate high-quality colorization, closely matching the original images.

---

## Frontend
The user interface is built with React and Vite, providing a modern, responsive, and user-friendly experience. Users can upload grayscale images, which are sent to the Flask backend for colorization. The results are displayed alongside the original and colorized images, allowing for easy comparison.

---

## Web App

- **Backend**: Flask
- **Frontend**: React with Vite
- **Functionality**:
  - Users upload grayscale images via the React frontend.
  - The Flask backend colorizes the images using the trained model.
  - Uploaded images and results are temporarily saved and automatically deleted after 60 seconds to manage storage.
  - The frontend communicates with the backend via REST API.

---

## Repository Structure

```
Space Image Colorizer/
├── api/                  # Flask web app backend
│   ├── static/           # Static files (uploads, results)
│   └── app.py            # Flask application code
├── frontend/             # React frontend with Vite
│   ├── src/              # React source files
│   ├── public/           # React public files
│   └── package.json      # Frontend dependencies
├── models/               # Model definition
│   └── colorize_unet.py
├── scripts/              # Training scripts
├── requirements.txt      # Python dependencies
├── README.md             # This file
```

---

**Image Source:**  
[NASA Image and Video Library](https://images.nasa.gov/)
