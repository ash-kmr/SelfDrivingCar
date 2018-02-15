import os
import cv2
import math
import numpy as np
from grabscreen import grab_screen

def draw_linear_regression_line(coef, intercept, intersection_x, img, imshape=[540,960], color=[255, 0, 0], thickness=2):
    point_one = (int(intersection_x), int(intersection_x * coef + intercept))
    if coef > 0:
        point_two = (imshape[1], int(imshape[1] * coef + intercept))
    elif coef < 0:
        point_two = (0, int(0 * coef + intercept))
    print("Point one: ", point_one, "Point two: ", point_two)
    cv2.line(img, point_one, point_two, color, thickness)
    return img

def linefind(slope_intercept):
    kept_slopes = []
    kept_intercepts = []
    print("Slope & intercept: ", slope_intercept)
    if len(slope_intercept) == 1:
        return slope_intercept[0][0], slope_intercept[0][1]
    slopes = [pair[0] for pair in slope_intercept]
    mean_slope = np.mean(slopes)
    slope_std = np.std(slopes)
    for pair in slope_intercept:
        slope = pair[0]
        if slope - mean_slope < 1.5 * slope_std:
            kept_slopes.append(slope)
            kept_intercepts.append(pair[1])
    if not kept_slopes:
        kept_slopes = slopes
        kept_intercepts = [pair[1] for pair in slope_intercept]
    slope = np.mean(kept_slopes)
    intercept = np.mean(kept_intercepts)
    print("Slope: ", slope, "Intercept: ", intercept)
    return slope, intercept

def draw_lanes(new_img, lines_img):
	imshape = [800, 600]
	positive_slope_points = []
	negative_slope_points = []
	positive_slope_intercept = []
	negative_slope_intercept = []
	for l in lines_img:
		for x1, y1, x2, y2 in l:
			m = (y1-y2)/(x1-x2)
			length = math.sqrt((x1-x2)**2 + (y1-y2)**2)
			if not math.isnan(m) and length > 40:
				if m > 0:
					positive_slope_points.append([x1, y1])
					positive_slope_points.append([x2, y2])
					positive_slope_intercept.append([m, y1-m*x1])
				elif m < 0:
					negative_slope_points.append([x1, y1])
					negative_slope_points.append([x2, y2])
					negative_slope_intercept.append([m, y1-m*x1])
	if not positive_slope_points:
		for line in lines_img:
			for x1,y1,x2,y2 in line:            
				slope = (y1-y2)/(x1-x2)
				if slope > 0:
					positive_slope_points.append([x1, y1])
					positive_slope_points.append([x2, y2])
					positive_slope_intercept.append([m, y1-m*x1])
	if not negative_slope_points:
		for line in lines_img:
			for x1,y1,x2,y2 in line:            
				slope = (y1-y2)/(x1-x2)
				if slope < 0:
					negative_slope_points.append([x1, y1])
					negative_slope_points.append([x2, y2])
					negative_slope_intercept.append([m, y1-m*x1])

	pcof, pint = linefind(positive_slope_intercept)
	ncof, nint = linefind(negative_slope_intercept)
	x = (pint - nint)/(ncof - pcof)
	draw_lrline(pcof, pint, x, new_img)
	draw_lrline(ncof, nint, x, new_img)
	return new_img

def draw_lrline(coef, intercept, intersection_x, img, imshape=[800,600], color=[255, 0, 0], thickness=2):
	try:
		print("Coef: ", coef, "Intercept: ", intercept, "intersection_x: ", intersection_x)
		point_one = (int(intersection_x), int(intersection_x * coef + intercept))
		if coef > 0:
			point_two = (imshape[1], int(imshape[1] * coef + intercept))
		elif coef < 0:
			point_two = (0, int(0 * coef + intercept))
		print("Point one: ", point_one, "Point two: ", point_two)
		cv2.line(img, point_one, point_two, color, thickness)
	except Exception as e:
		print(str(e))

def lane_detect(image):
	"""l_yellow = np.array([20, 100, 100], dtype = np.uint8)
	u_yellow = np.array([30, 255, 255], dtype=np.uint8)
	mask_y = cv2.inRange(hsv, l_yellow, u_yellow)
	mask_w = cv2.inRange(gray_image, 200, 255)
	mask_or = cv2.bitwise_or(mask_w, mask_y)
	image = cv2.bitwise_and(image, mask_or)"""
	image = cv2.GaussianBlur(image, (5, 5), 0)
	image = cv2.Canny(image, 50, 150)
	## masking
	mask_vertices = np.array([[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask, [mask_vertices], 255)
	image = cv2.bitwise_and(image, mask)
	## getting hough lines
	lines_image = cv2.HoughLinesP(image, 2, np.pi/180, 45, np.array([]), 40, 100)
	new_img = np.zeros(image.shape, dtype = np.uint8)
	new_img = draw_lanes(new_img, lines_image)
	return new_img

def main():
		screen = grab_screen(region = (0, 40, 800, 600))
		screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
		screen = lane_detect(screen)
		cv2.imshow("window", screen)
		cv2.waitKey()

main()