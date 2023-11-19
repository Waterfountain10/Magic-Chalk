from model import test_images, test_labels
from tensorflow.keras.models import load_model

loaded_model = load_model('my_digit_symbol_model.keras')
test_loss, test_accuracy = loaded_model.evaluate(test_images, test_labels)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)