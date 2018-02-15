import numpy as np
from PIL import ImageGrab
import cv2
import time
from keyboard import PressKey, ReleaseKey, W, A, S, D
from draw_lines import draw_lanes
from grabscreen import 	grab_screen
from getkeys import key_check
import os
import keras
"""
filename = "train.npy"
if os.path.isfile(filename):
	print("exists")
	train = list(np.load(filename))
else:
	print("not found")
	train = []
"""



def reg_of_int(img, vertices):
	mask = np.zeros_like(img)
	cv2.fillPoly(mask, vertices, 255)
	mask_app = cv2.bitwise_and(img, mask)
	return mask_app

def keys_to_outputs(keys):
	#[A, W, D]
	out = [0, 0, 0]
	if 'A' in keys:
		out[0] = 1
	elif 'D' in keys:
		out[2] = 1
	else:
		out[1] = 1
	return out

def process(image):
	l_yellow = np.array([20, 100, 100], dtype = np.uint8)
	u_yellow = np.array([30, 255, 255], dtype=np.uint8)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	mask_y = cv2.inRange(hsv, l_yellow, u_yellow)
	mask_w = cv2.inRange(gray_image, 200, 255)
	mask_or = cv2.bitwise_or(mask_w, mask_y)
	image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img_for_processing = cv2.bitwise_and(image2, mask_or)
	img_for_processing = cv2.GaussianBlur(img_for_processing, (5, 5), 0)
	img_for_processing = cv2.Canny(image, threshold1 = 50, threshold2 = 150)
	vertices = np.array([[10, 500], [10, 250], [300, 200], [500, 200], [800, 250], [800, 500], [750, 500], [400, 230], [60, 500]])
	processed_img = reg_of_int(img_for_processing, [vertices])
	#cv2.imshow("wid", processed_img)
	#cv2.waitKey()
	#return processed_img
	lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180,np.array([]), 	60, 5)
	m1 = 0
	m2 = 0
	try:
		l1, l2, m1, m2 = draw_lanes(image,lines)
		cv2.line(image, (l1[0], l1[1]), (l1[2], l1[3]), [0,123,134], 30)
		cv2.line(image, (l2[0], l2[1]), (l2[2], l2[3]), [0,123,134], 30)
	except Exception as e:
		print(str(e))
		print("under exception")
		pass

	return image, m1, m2

def Straight():
	PressKey(W)
	ReleaseKey(A)
	ReleaseKey(D)

def Left():
	PressKey(A)
	ReleaseKey(W)
	ReleaseKey(D)
	time.sleep(0.01)
	ReleaseKey(A)

def Right():
	PressKey(D)
	ReleaseKey(A)
	ReleaseKey(W)
	time.sleep(0.01)
	ReleaseKey(D)

def Slow():
	PressKey(S)
	ReleaseKey(W)
	ReleaseKey(A)
	time.sleep(0.05)
	ReleaseKey(S)

def main():
	model = keras.models.load_model("weights_2.hdf5")
	i = 0
	print("model loaded")
	while(True):
		i = i + 1
		if i%2 == 0: Straight()
		screen = grab_screen(region = (0, 40, 750, 550))
		new_sc = screen
		new_sc = cv2.cvtColor(new_sc, cv2.COLOR_BGR2GRAY)
		new_sc = cv2.resize(new_sc, (80, 60))
		new_sc = np.array(new_sc).reshape(1, 80, 60, 1)
		"""
		screen = cv2.resize(screen, (160, 120))
		keys = key_check()
		output = keys_to_outputs(keys)
		train.append([screen, output])
		if len(train) % 1000 == 0:
			print("1000")
		if len(train) % 7000 == 0:
			print("SAVING DATA")
			np.save(filename, train)
			print("saved")"""
		processed_img, m1, m2 = process(screen)
		if m1 < 0 and m2 < 0:
			if abs(m1) > 0.3 and abs(m2) > 0.3:
				print(m1 , " " , m2)
				print("right")
				Right()
			else:
				a = model.predict(new_sc)
				b = a[0].argmax()
				print("model")
				if b == 0:
					Left()
					print("left")
				elif b == 2:
					Right()
					print("right")
				else:
					Straight()
					print("straight")

		elif m1 > 0 and m2 > 0:
			if abs(m1) > 0.3 and abs(m2) > 0.3:
				print(m1 , " " , m2)
				print("left")
				Left()
			else:
				a = model.predict(new_sc)
				b = a[0].argmax()
				print("model")
				if b == 0:
					Left()
					print("left")
				elif b == 2:
					Right()
					print("right")
				else:
					Straight()
					print("straight")

		elif abs(m1) > 0: 
			print(m1 , " " , m2)
			Straight()

		else:
			print("model")
			a = model.predict(new_sc)
			b = a[0].argmax()
			if b == 0:
				Left()
				print("left")
			elif b == 2:
				Right()
				print("right")
			else:
				Straight()
				print("straight")
		cv2.imshow('window',cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

if __name__=='__main__':
	main()