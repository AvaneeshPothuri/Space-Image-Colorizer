import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [colorizedImage, setColorizedImage] = useState(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select an image to upload.');
      return;
    }

    setError('');
    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('https://space-image-colorizer.onrender.com/api/colorize', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const uploadedUrl = `https://space-image-colorizer.onrender.com${response.data.upload}`;
      const colorizedUrl = `https://space-image-colorizer.onrender.com${response.data.result}`;

      setUploadedImage(uploadedUrl);
      setColorizedImage(colorizedUrl);
    } catch (err) {
      console.error('Error uploading image:', err);
      setError('Failed to upload and colorize image.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Space Image Colorizer</h1>
        <p>Upload a grayscale space image to see it colorized by AI</p>
      </header>
      <main className="main">
        <form onSubmit={handleUpload} className="upload-form">
          <label className="file-input-label">
            <span>Choose Image</span>
            <input
              type="file"
              accept="image/png, image/jpeg"
              onChange={(e) => setFile(e.target.files[0])}
              className="file-input"
            />
          </label>
          <button type="submit" className="upload-button" disabled={isLoading}>
            {isLoading ? 'Processing...' : 'Upload and Colorize'}
          </button>
          {error && <p className="error-message">{error}</p>}
        </form>
        <div className="image-display">
          {uploadedImage && (
            <div className="image-card">
              <h3>Original</h3>
              <img src={uploadedImage} alt="Uploaded" />
            </div>
          )}
          {colorizedImage && (
            <div className="image-card">
              <h3>Colorized</h3>
              <img src={colorizedImage} alt="Colorized" />
            </div>
          )}
        </div>
      </main>
      <footer className="footer">
        <p>© 2025 Space Image Colorizer</p>
      </footer>
    </div>
  );
}

export default App;