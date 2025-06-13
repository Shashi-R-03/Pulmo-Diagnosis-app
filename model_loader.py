import tensorflow as tf

def load_models():
    # Load the previously trained ResNet model
    model = tf.keras.models.load_model('resnet_chest_xray_classifier.h5')
    return model

# Example usage
if __name__ == "__main__":
    model = load_models()
    print("Model loaded successfully.")