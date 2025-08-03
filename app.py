import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Define the path to the Keras model
# Ensure 'mnist.keras' is in the same directory as this app.py file
MODEL_PATH = 'mnist.keras'

# Load the Keras model
# It's crucial to load the model only once when the app starts
# to avoid reloading it on every request, which would be very slow.
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

@app.route('/')
def index():
    """
    Renders the main HTML page with the drawing canvas.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives image data from the frontend, preprocesses it,
    and makes a prediction using the loaded Keras model.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    try:
        # Get the base64 encoded image data from the request
        data = request.get_json()
        img_data_base64 = data['image'].split(',')[1] # Remove "data:image/png;base64," prefix

        # Decode base64 to bytes
        img_bytes = base64.b64decode(img_data_base64)

        # Open the image using Pillow (PIL)
        img = Image.open(io.BytesIO(img_bytes))

        # Preprocess the image for the model
        # The MNIST model expects 28x28 grayscale images, normalized to 0-1
        img = img.resize((28, 28)) # Resize to 28x28
        img = img.convert('L')     # Convert to grayscale ('L' mode)
        img_array = np.array(img)  # Convert to NumPy array

        # Invert colors if necessary (MNIST typically has white digits on black background).
        # Your current drawing on the frontend is white on black.
        # If your model was trained on white digits on a black background,
        # DO NOT invert the colors here.
        # If your model was trained on BLACK digits on a WHITE background,
        # then you WOULD need to invert by uncommenting the line below.
        # img_array = 255 - img_array # Uncomment ONLY if your model expects black digits on white background

        img_array = img_array / 255.0 # Normalize pixel values to 0-1

        # Reshape the image for the model: (batch_size, height, width)
        # For a single image, batch_size is 1, so shape becomes (1, 28, 28)
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_digit = np.argmax(predictions[0]) # Get the digit with the highest probability

        # Return the prediction as a JSON response
        return jsonify({'prediction': int(predicted_digit)})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure the 'templates' directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')

    # This is for local development. In a production environment,
    # you would typically use a WSGI server (e.g., Gunicorn, uWSGI).
    app.run(debug=True, host='0.0.0.0', port=5000)
