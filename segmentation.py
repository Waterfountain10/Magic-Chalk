import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('my_digit_symbol_model.keras')


def segment_and_predict(image_path = 'mask_capture.jpeg'):
    # Read the image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding for black text on white background
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Invert the thresholded image for dilation (since text is now black)
    inverted_thresh = cv2.bitwise_not(thresh)

    # Apply dilation to close gaps in characters
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(inverted_thresh, kernel, iterations=1)

    # Find contours on the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    predictions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Crop from the original thresholded image (not the dilated one)
        cropped_char = thresh[y:y+h, x:x+w]
        resized_char = cv2.resize(cropped_char, (28, 28))

        normalized_char = resized_char / 255.0
        normalized_char = normalized_char.reshape(-1, 28, 28, 1)

        prediction = model.predict(normalized_char)
        predicted_class = np.argmax(prediction, axis=-1)
        predictions.append(predicted_class[0])

        # Visualization with prediction
        #cv2.imshow(f"Predicted: {predicted_class[0]}", resized_char)
        #cv2.waitKey(0)

    #cv2.destroyAllWindows()
    predictions[0], predictions[1] = predictions[1], predictions[0]

    return predictions



#result = segment_and_predict()
#print("Predicted Characters:", result)