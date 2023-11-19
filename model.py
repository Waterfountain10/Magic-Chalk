# Import necessary libraries
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

# Define the directory
data_dir = r"C:\Users\kevin\Desktop\codejam\dataset"

# Load and preprocess images
def load_images(directory):
    images = []
    labels = []
    for label_folder in os.listdir(directory):
        label_folder_path = os.path.join(directory, label_folder)
        for image_file in os.listdir(label_folder_path):
            image_path = os.path.join(label_folder_path, image_file)
            image = Image.open(image_path).convert('L') # Convert to grayscale
            image = image.resize((28, 28)) # Resize image
            images.append(np.array(image))
            labels.append(label_folder)
    return np.array(images), np.array(labels)

# Loading images and labels
images, labels = load_images(data_dir)

# Normalize pixel values
images = images / 255.0

# Split the data
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)


from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Convert labels to numerical format
label_encoder = LabelEncoder()
numerical_labels = label_encoder.fit_transform(train_labels)

# One-hot encode the labels
one_hot_labels = to_categorical(numerical_labels, num_classes=15)  # Assuming 15 different classes

# Also process the test labels
test_numerical_labels = label_encoder.transform(test_labels)
test_one_hot_labels = to_categorical(test_numerical_labels, num_classes=15)

# Now, train_labels and test_labels are one-hot encoded
train_labels = one_hot_labels
test_labels = test_one_hot_labels

# Reshape data to fit the model
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Initialize the model
model = Sequential()

# Add model layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='softmax'))  # 15 classes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_images, train_labels,
    epochs=60,  # Number of epochs
    batch_size=32,  # Batch size
    validation_data=(test_images, test_labels)
)

# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# Saving the model
model.save('my_digit_symbol_model.keras')

