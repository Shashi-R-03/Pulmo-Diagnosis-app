from flask import Flask, request, render_template, jsonify
from model_loader import load_models
from data_preprocessing import preprocess_image
from grad_cam import generate_grad_cam, superimpose_heatmap
import cv2
import numpy as np
import cohere

app = Flask(__name__)
model = load_models()  # Load the model once when the app starts

# Initialize Cohere API
cohere_api_key = 'YOUR_API_KEY'  # Replace with your actual API key
co = cohere.Client(cohere_api_key)

# Define class names
class_names = {
    0: "COVID",
    1: "Tuberculosis",
    2: "Pneumonia",
    3: "Normal"
}

@app.route('/')
def home():
    return render_template('index.html')  # Render the home page

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Process the uploaded image
    image_path = f"static/uploads/{file.filename}"
    file.save(image_path)
    processed_image = preprocess_image(image_path)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]  # Get class name
    confidence_scores = prediction[0]  # Get confidence scores for all classes

    # Generate Grad-CAM
    heatmap = generate_grad_cam(model, processed_image, predicted_class_index)
    original_image = cv2.imread(image_path)
    superimposed_image = superimpose_heatmap(heatmap, original_image)

    # Save the superimposed image
    superimposed_image_path = f"static/uploads/superimposed_{file.filename}"
    cv2.imwrite(superimposed_image_path, superimposed_image)

    return render_template('result.html', 
                           predicted_class=predicted_class_name, 
                           confidence_scores=confidence_scores,
                           class_names=class_names,  # Pass class names to the template
                           superimposed_image=superimposed_image_path)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = co.generate(
        model='command',  # Use a valid model ID
        prompt=f"Answer the following question about pulmonary diseases: {user_input}",
        max_tokens=10,
        temperature=0.5
    )
    return jsonify({'response': response.generations[0].text.strip()})

if __name__ == '__main__':
    app.run(debug=True)
