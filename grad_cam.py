import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model_loader import load_models
from data_preprocessing import preprocess_image

def generate_grad_cam(model, image, class_index):
    print("In generate_grad_cam function")
    
    # Get the last convolutional layer
    print("Getting last convolutional layer...")
    try:
        last_conv_layer = model.get_layer('conv5_block3_out')
        print(f"Last conv layer found: {last_conv_layer.name}")
    except Exception as e:
        print(f"Error getting layer: {e}")
        print("Available layers:", [layer.name for layer in model.layers])
        raise
    
    # Create a model that outputs the last conv layer and the output layer
    print("Creating grad model...")
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.input,  # Remove the list brackets
            outputs=[model.get_layer('dense').output, last_conv_layer.output]  # Reverse the order here
        )
        print("Grad model created successfully")
    except Exception as e:
        print(f"Error creating grad model: {e}")
        raise
    
    print("Running gradient tape...")
    with tf.GradientTape() as tape:
        # Ensure the input is wrapped in a tensor
        print("Converting input to tensor...")
        tensor_input = tf.convert_to_tensor(image, dtype=tf.float32)
        print(f"Input tensor shape: {tensor_input.shape}")
        
        print("Getting model output and conv layer output...")
        class_output, conv_output = grad_model(tensor_input)  # Renamed variables for clarity
        print(f"Class output shape: {class_output.shape}")
        print(f"Conv output shape: {conv_output.shape}")
        
        loss = class_output[:, class_index]
        print(f"Loss shape: {loss.shape}")
    
    # Compute gradients
    print("Computing gradients...")
    # Watch the variable to ensure gradients are computed
    tape.watch(conv_output)
    grads = tape.gradient(loss, conv_output)
    print(f"Gradients shape: {grads.shape if grads is not None else 'None'}")

    # Handle None gradients
    if grads is None:
        print("Warning: Gradients are None. Using a workaround...")
        # Use a different approach or provide a meaningful error message
        # For now, we'll create dummy gradients to continue
        grads = tf.zeros_like(conv_output)
    
    # Check the shape of conv_output
    if len(conv_output.shape) == 4:  # If it's a 4D tensor
        print("Conv layer output is a 4D tensor, computing pooled gradients...")
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        print(f"Pooled gradients shape: {pooled_grads.shape}")
    else:
        error_msg = f"Unexpected shape for conv_output: {conv_output.shape}"
        print(error_msg)
        raise ValueError(error_msg)

    # Create a heatmap
    print("Creating heatmap...")
    heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    print(f"Heatmap shape: {heatmap.shape}")

    # Normalize the heatmap
    print("Normalizing heatmap...")
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= tf.reduce_max(heatmap)  # Normalize
    print("Heatmap normalized")

    return heatmap.numpy()

def generate_grad_cam_simple(model, image, class_index):
    # Create a model that maps the input directly to the activations of the last conv layer
    last_conv_layer = model.get_layer('conv5_block3_out')
    
    # Get the gradients of the predicted class with respect to the last conv layer
    with tf.GradientTape() as tape:
        # Ensure inputs are tensors
        inputs = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Get the conv layer output
        last_conv_output = last_conv_layer(inputs, training=False)
        tape.watch(last_conv_output)
        
        # Get the prediction
        preds = model(inputs, training=False)
        pred_index = class_index
        class_channel = preds[:, pred_index]
    
    # Compute gradients
    grads = tape.gradient(class_channel, last_conv_output)
    
    # Apply global average pooling to the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map by its importance
    last_conv_output = last_conv_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    # Weight the channels by the gradients
    for i in range(pooled_grads.shape[0]):
        last_conv_output[:, :, i] *= pooled_grads[i]
    
    # Average over all the channels
    heatmap = np.mean(last_conv_output, axis=-1)
    
    # ReLU and normalize
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    return heatmap

def superimpose_heatmap(heatmap, original_image, alpha=0.4):
    # Resize heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # Scale to 0-255
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply color map

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(original_image, alpha, heatmap, 1 - alpha, 0)
    return superimposed_img

# Example usage
if __name__ == "__main__":
    print("Starting Grad-CAM script...")
    try:
        model = load_models()
        print("Model loaded successfully")
        
        print("Printing model summary:")
        model.summary()
        
        image_path = 'C:/Users/dasta/OneDrive/Desktop/model/chest_xray_classifier/sample images/NEUMONIA/person20_virus_51.jpeg'
        print(f"Processing image from: {image_path}")
        
        processed_image = preprocess_image(image_path)
        print("Image preprocessed successfully")
        
        # Generate Grad-CAM for the predicted class
        print("Getting model prediction...")
        prediction = model.predict(processed_image)
        print(f"Prediction shape: {prediction.shape}")
        
        predicted_class = np.argmax(prediction, axis=1)[0]
        print(f"Predicted class: {predicted_class}")
        
        print("Generating Grad-CAM...")
        heatmap = generate_grad_cam(model, processed_image, predicted_class)
        print("Grad-CAM generated successfully")
        
        # Superimpose heatmap on the original image
        original_image = cv2.imread(image_path)
        print("Original image loaded")
        
        superimposed_image = superimpose_heatmap(heatmap, original_image)
        print("Heatmap superimposed on original image")
        
        # Display the result
        plt.imshow(superimposed_image)
        plt.axis('off')
        plt.show()
        print("Image displayed")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()