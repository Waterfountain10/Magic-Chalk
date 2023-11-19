# Magic Chalk - Interactive Whiteboard Application

## Description
Magic Chalk is an interactive whiteboard application developed using Python. It employs technologies like Streamlit for the web interface, OpenCV for image processing, and MediaPipe for hand gesture recognition. This application allows users to draw, erase, solve mathematical equations, and save their work with hand gestures.

## Installation & Dependencies
Before running the application, ensure that you have the following libraries installed:
- Streamlit
- OpenCV
- MediaPipe
- NumPy
- WolframAlpha API
- Tensorflow
- Scikitlearn

You can install them using pip:
```bash
pip install streamlit opencv-python mediapipe numpy wolframalpha tensorflow scikitlearn
```

## Usage
To start the application, run the following command in your terminal:
```bash
streamlit run your_script_name.py
```
### Available Tools
![alt text](tools.png) <br>
Draw - Erase - Clear - Solve - Bookmark

### How to draw
- Raise the index to select tool <br>
- Raise index and middle finger to draw or erase

## Note
- Ensure your camera is properly configured and accessible
  - Verify `cap = cv2.VideoCapture(0)`
- The application uses a webcam with specific resolution settings
- Gesture recognition may vary based on lighting conditions
