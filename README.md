# Magic Chalk - Interactive Whiteboard Application

Magic Chalk is an interactive whiteboard application developed using Python. It employs technologies like Streamlit for the web interface, OpenCV for image processing, and MediaPipe for hand gesture recognition. This application allows users to draw, erase, solve mathematical equations, and save their work with hand gestures.

## Installation & Dependencies
Before running the application, ensure that you have the following libraries installed:
- Streamlit
- OpenCV
- MediaPipe
- NumPy
- WolframAlpha API
  - Make sure you have an API key

You can install them using pip:
```bash
pip install streamlit opencv-python mediapipe numpy wolframalpha
```

## Train Model
You can download this [dataset](https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols/code) to train your model
- There is already a model that you can use to train, however you might need to install other dependencies

#### N.B.
- Only keep 0-9, + - ÷ x
- Delete .directory in the main folder and in folder 9

## Start & Usage
To start the application, run the following command in your terminal:
```bash
streamlit run main.py
```
### Available Tools
![tools](tools.png) <br>
Draw - Erase - Clear - Solve - Bookmark

### How to draw
- Raise the index to select tool <br>
- Raise index and middle finger to draw or erase

## Note
- Ensure your camera is properly configured and accessible
  - Verify `cap = cv2.VideoCapture(0)`
- The application uses a webcam with specific resolution settings
- Gesture recognition may vary based on lighting conditions
