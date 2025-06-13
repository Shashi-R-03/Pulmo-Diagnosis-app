# Pulmo Diagnosis App

Pulmo Diagnosis App is a deep-learning-powered web application for analyzing chest X-ray images to predict pulmonary diseases. It uses a pre-trained Convolutional Neural Network (CNN) to classify each X-ray into one of four categories: **COVID-19**, **Tuberculosis**, **Pneumonia**, or **Normal**. The model is loaded when the Flask server starts up. Users can upload an X-ray image through the web interface, and the app returns the predicted disease along with a Grad-CAM heatmap to interpret the model's prediction.

---

## 🚀 Features

* **X-ray Input**: Upload chest X-ray images via the browser.
* **CNN-Based Prediction**: Predicts one of COVID-19, TB, Pneumonia, or Normal.
* **Grad-CAM Visualization**: Heatmaps showing areas influencing prediction.
* **Flask Web Interface**: Simple frontend to upload and get predictions.
* **(Optional)** **LIME Explanations** and **Cohere-based Q\&A** chatbot.

---

## 🛠️ Tech Stack

* **Backend**: Python, Flask
* **Deep Learning**: CNN (via TensorFlow or PyTorch)
* **Image Processing**: OpenCV, NumPy
* **Explainability**: Grad-CAM, LIME
* **NLP API**: Cohere

---

## 📥 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Shashi-R-03/Pulmo-Diagnosis-app.git
cd Pulmo-Diagnosis-app
```

### 2. Install Dependencies

If `requirements.txt` is not available, install manually:

```bash
pip install flask opencv-python numpy scikit-image lime cohere
```

Also install your deep learning framework (e.g., TensorFlow or PyTorch):

```bash
pip install tensorflow  # or torch
```

### 3. Setup Cohere API (Optional)

If using the Q\&A feature, insert your API key in `app.py`:

```python
cohere_api_key = "your_actual_key"
```

### 4. Run the Flask App

```bash
python app.py
```

App will run on: `http://127.0.0.1:5000/`

---

## 🧪 Usage Guide

1. **Open the App**: Navigate to `http://127.0.0.1:5000/`
2. **Upload an Image**: Choose and upload a chest X-ray image (JPG/PNG).
3. **Click Predict**: The image is analyzed and result displayed.
4. **View Results**:

   * Predicted class (COVID, TB, Pneumonia, Normal)
   * Grad-CAM heatmap showing focus regions
5. **(Optional)** Ask medical questions in the chatbot interface

---

## 🚀 Deployment (Optional)

To deploy on platforms like Heroku:

1. Add a `Procfile`:

```
web: gunicorn app:app
```

2. Use Gunicorn for production:

```bash
pip install gunicorn
```

3. Push to a hosting platform with proper config.

---

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 Acknowledgements

* CNN architecture inspiration
* Grad-CAM and LIME explainability techniques
* [Cohere](https://cohere.ai) for language-based Q\&A
