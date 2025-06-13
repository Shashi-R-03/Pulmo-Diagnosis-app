import cv2
import numpy as np

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Resize the image to the input size expected by the models (e.g., 224x224)
    image = cv2.resize(image, (224, 224))
    
    # Normalize the image
    image = image.astype('float32') / 255.0
    
    # Expand dimensions to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    
    return image

# Example usage
if __name__ == "__main__":
    processed_image = preprocess_image('C:/Users/dasta/OneDrive/Desktop/model/chest_xray_classifier/sample images/NEUMONIA/person14_virus_44.jpeg')
    print("Image preprocessed successfully.")

