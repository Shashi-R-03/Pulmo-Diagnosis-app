from flask import Flask, request, render_template, jsonify, url_for
from model_loader import load_models
from data_preprocessing import preprocess_image
from grad_cam import generate_grad_cam, superimpose_heatmap
import cv2
import numpy as np
import cohere
from lime import lime_image
from skimage.segmentation import mark_boundaries

app = Flask(__name__)
model = load_models()  # Load the model once when the app starts

# Initialize Cohere API
cohere_api_key = 'DbqJjall1lDsQewj14wfw5uulIM7DYqKF5CsgJDA'  # Replace with your actual API key
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
                           superimposed_image=superimposed_image_path,
                           file=file)  # Pass the file variable to the template

def generate_lime_explanation(model, image):
    # Create a LIME image explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Define a function to wrap the model's predict method
    def model_predict_fn(images):
        return model.predict(images)

    # Generate LIME explanation
    explanation = explainer.explain_instance(
        image[0],  # Use the first image in the batch
        model_predict_fn,  # Use the wrapped model predict function
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    
    # Get the image and mask for the top label
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        hide_rest=False,
        num_features=10
    )
    
    # Load the original image
    original_image = (image[0] * 255).astype(np.uint8)  # Convert to uint8 for overlay
    marked_image = mark_boundaries(temp / 255.0, mask)

    # Ensure marked_image is also uint8
    marked_image = (marked_image * 255).astype(np.uint8)  # Convert to uint8 for overlay

    # Overlay the LIME explanation on the original image
    overlay_image = cv2.addWeighted(original_image, 0.5, marked_image, 0.7, 0)  # Adjust weights as needed

    # Save the overlay image to a file
    marked_image_path = 'static/uploads/lime_explanation.png'
    cv2.imwrite(marked_image_path, overlay_image)  # Save as an image

    return marked_image_path  # Return the path for display

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = co.generate(
        model='command',  # Use a valid model ID
        prompt=f"Answer the following question about pulmonary diseases: {user_input}",
        max_tokens=40,
        temperature=0.5
    )
    return jsonify({'response': response.generations[0].text.strip()})

@app.route('/generate_lime', methods=['POST'])
def generate_lime():
    data = request.get_json()
    filename = data.get('filename')
    image_path = f"static/uploads/{filename}"

    # Load the processed image
    processed_image = preprocess_image(image_path)

    # Generate LIME explanation
    lime_explanation_path = generate_lime_explanation(model, processed_image)

    return jsonify({'success': True, 'lime_image_path': url_for('static', filename='uploads/lime_explanation.png')})

if __name__ == '__main__':
    app.run(debug=True)