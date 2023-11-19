import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from wolfram_calculator import compute_latex_expression #set api key 
from segmentation import segment_and_predict

st.set_page_config(
  page_title="CodeJam", 
  layout="wide",
  initial_sidebar_state="expanded",
)


# frame_placeholder = st.empty()
camera_view_placeholder = st.empty()


def open_camera(camera_view_placeholder):

	#contants
	margin_left = 150
	max_x, max_y = 250 + margin_left, 50
	curr_tool = "select tool"
	start_time = True
	rad = 30
	thick = 8
	prevx, prevy = 0,0

	def save_mask_as_image(mask, file_format='jpeg'):
		filename = f'mask_capture.{file_format}'
		if file_format == 'png':
			cv2.imwrite(filename, mask)
		else:
			cv2.imwrite(filename, mask, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
		
	#get tools function
	def get_tool(x):
		if x < 50 + margin_left:
			return "draw"

		elif x < 100 + margin_left:
			return "erase"

		elif x < 150 + margin_left:
			return "clear"

		elif x < 200 + margin_left:
			return "solve"

		else:
			return "save"

	#y corresponds to the landmarks of the middle finger's landmark
	def middle_finger_raised(yi, y9):
		return (y9 - yi) > 40

	def display_numbers(list_numbers):
		int_to_str = {
			'0': '0',
			'1': '1',
			'2': '2',
			'3': '3',
			'4': '4',
			'5': '5',
			'6': '6',
			'7': '7',
			'8': '8',
			'9': '9',
			'10': "+",
			'11': "/",
			'12': "=",
			'13': '*',
			'14': '-'
			
			}
		output = ""

		if len(list_numbers) > 2:
			contains_operator = False

			for i in list_numbers:
				if int(i) > 9:
					contains_operator = True

			try: #Avoid error message
				if contains_operator:
					operator = max(list_numbers)
					index = list_numbers.index(operator) #index of operator
					if index == 1:
						for key in list_numbers:
							output += int_to_str[str(key)]
					else:
						list_numbers[index], list_numbers[1] = list_numbers[1], list_numbers[index]
						for key in list_numbers:
							output += int_to_str[str(key)]

				return output
			
			except Exception:
				return output

		else:  # 1 or 2 characters only
			for key in list_numbers:
				output += int_to_str[str(key)]

	hands = mp.solutions.hands
	hand_landmark = hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1)
	draw = mp.solutions.drawing_utils

	tools = cv2.imread("tools.png")
	tools = tools.astype('uint8')

	mask = np.ones((480, 640))*255
	mask = mask.astype('uint8')

	cap = cv2.VideoCapture(0)
	while True:
		_, frm = cap.read()
		frm = cv2.flip(frm, 1)

		rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

		op = hand_landmark.process(rgb)

		if not _:
			st.write("ended")
			break


		if op.multi_hand_landmarks:
			for i in op.multi_hand_landmarks:
				draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
				x, y = int(i.landmark[8].x*640), int(i.landmark[8].y*480)

				if x < max_x and y < max_y and x > margin_left:
					if start_time:
						ctime = time.time()
						start_time = False
					ptime = time.time()

					cv2.circle(frm, (x, y), rad, (0 , 255, 0), 2)
					rad -= 1

					if (ptime - ctime) > 0.8:
						curr_tool = get_tool(x)
						print("your current tool set to : ", curr_tool)
						start_time = True
						rad = 30

				else:
					start_time = True
					rad = 30

				if curr_tool == "draw":
					xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
					y9  = int(i.landmark[9].y * 480)

					if middle_finger_raised(yi, y9):
						cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
						prevx, prevy = x, y

					else:
						prevx = x
						prevy = y

				elif curr_tool == "erase":
					xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
					y9  = int(i.landmark[9].y*480)

					if middle_finger_raised(yi, y9):
						cv2.circle(frm, (x, y), 30, (0,0,0), -1)
						cv2.circle(mask, (x, y), 30, 255, -1)
				
				elif curr_tool == 'solve':

					save_mask_as_image(mask, 'jpeg')
					list_numbers = segment_and_predict()
					output = display_numbers(list_numbers)

					answer = compute_latex_expression(output, 'QY6LX3-5UPVEGR9Y9')

					st.sidebar.latex(f"{output} \quad {answer}")
					curr_tool = 'solved'

				elif curr_tool == 'save':
					save_mask_as_image(mask, 'jpeg')
					
					list_numbers = segment_and_predict() #stores as a list of integer
					
					output = display_numbers(list_numbers)
					st.sidebar.latex(output) 
					
					curr_tool = 'saved'

					
				
				elif curr_tool == 'clear':
					mask.fill(255)
					

		
		op = cv2.bitwise_and(frm, frm, mask=mask)
		frm[:, :, 2] = op[:, :, 2]
		frm[:, :, 1] = op[:, :, 1]

		frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

		frm[:max_y, margin_left:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, margin_left:max_x], 0.3, 0)

		cv2.putText(frm, curr_tool, (270 + margin_left, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (35, 28, 221), 1)
		
		# frame_placeholder.image(frm)
		camera_view_placeholder.image(frm) # TEST

		if cv2.waitKey(1) == 27:
			cv2.destroyAllWindows()
			cap.release()
			break

##############
#MAIN PROGRAM#
##############

def main():
	col1, col2, col3 = st.columns([1, 5, 2])
	with col1:
		cs_sidebar()
	
	with col2:
		cs_body()
		
	with col3:  
		pics, captions = st.columns([1,1.8])

		with pics:
			info_pics()
		with captions:
			info_captions()
	

def cs_sidebar():
		st.sidebar.markdown('<div style="font-size:50px;">Magic Chalk</div>', unsafe_allow_html=True)
		st.sidebar.code('Transcript')

		return None

def info_pics():
	st.image('Instructions_pics/draw.png')
	st.image('Instructions_pics/erase.png')
	st.image('Instructions_pics/clear.png')
	st.image('Instructions_pics/solution.png')
	st.image('Instructions_pics/bookmark.png')


def info_captions():
	st.subheader('Draw Tool')
	st.subheader("")
	st.subheader('Erase Tool')
	st.subheader("")
	st.subheader('Clear Canvas')
	st.subheader("")
	st.subheader('Solve The Equation')
	st.subheader("")
	st.subheader('Bookmark')


def cs_body():

	#col1, col2 = st.columns([2,1])

	camera_view_placeholder = st.empty() # TEST
	open_camera_button = st.button("Open camera", type="primary")
	stop_button_pressed = st.button("Stop")

	if open_camera_button:
		# OPEN THE CAMERA
		open_camera(camera_view_placeholder)  # TEST

	st.markdown("Made by William Kiem Lafond, Kevin Liu, Orlando Qiu, and David Zhou")



if __name__ == '__main__':
	main()


hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html = True)