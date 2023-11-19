from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

loaded_model = load_model('my_digit_symbol_model.keras')

# Function to preprocess and predict new images
def predict_image():
    image = Image.open(r'mask_capture.jpeg').convert('L')

    # pre process
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(-1, 28, 28, 1)
    prediction = loaded_model.predict(image_array)

    return np.argmax(prediction)

#result = predict_image()
#print("Predicted Label:", result)
