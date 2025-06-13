import numpy as np
from model_loader import load_models

def predict_with_model(model, image):
    # Get predictions from the model
    prediction = model.predict(image)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)
    
    return predicted_class

# Example usage
if __name__ == "__main__":
    from model_loader import load_models
    from data_preprocessing import preprocess_image

    model = load_models()  # Load the previously built model
    image_path = 'C:/Users/dasta/OneDrive/Desktop/model/chest_xray_classifier/sample images/NEUMONIA/person14_virus_44.jpeg'  # Replace with your image path
    processed_image = preprocess_image(image_path)

    prediction = predict_with_model(model, processed_image)
    print(f"Predicted class: {prediction}")