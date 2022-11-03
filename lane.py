# LANE DETECTION SYSTEM

import cv2
import numpy as np
import matplotlib.pyplot as plt

# image = cv2.imread('image_test.jpg')
image = cv2.imread('test_image.jpg')

# STEP 1: CONVERT TO GRAYSCALE
lane_image = np.copy(image)
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

# STEP 2: APPLY GAUSSIAN BLUR
blur = cv2.GaussianBlur(gray, (5,5), 0)

# STEP 3: APPLY CANNY EDGE DETECTION
canny = cv2.Canny(blur, 50, 150)

# cv2.imshow('result', canny)
# cv2.waitKey(0)

# STEP 4: APPLY REGION OF INTEREST
height = canny.shape[0]
width = canny.shape[1]

print(width, height)
# region_of_interest_vertices = [
#     (0, height),
#     (width, int(height/2)),
#     (width, height)
# ]

# make region_of_interest_verticies a rectangle
region_of_interest_vertices = [
    (0, height),
    # (int(4*width/5), int(height/5)),
    (int(width/2), 0),
    (width, 0),
    (width, height)
]

# outline the region of interest in blue
# cv2.line(lane_image, region_of_interest_vertices[0], region_of_interest_vertices[1], (255,0,0), 3)
# cv2.line(lane_image, region_of_interest_vertices[1], region_of_interest_vertices[2], (255,0,0), 3)
# cv2.line(lane_image, region_of_interest_vertices[2], region_of_interest_vertices[0], (255,0,0), 3)

# cv2.imshow('result', lane_image)
# cv2.waitKey(0)

# STEP 5: APPLY HOUGH TRANSFORM
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

cropped_image = region_of_interest(canny, np.array([region_of_interest_vertices], np.int32))

# cv2.imshow('result', cropped_image)
# cv2.waitKey(0)

blur_cropped = cv2.GaussianBlur(cropped_image, (15,15), 0)
# cv2.imshow('result', blur_cropped)
# cv2.waitKey(0)

# sharpen the image to make the lines more visible
thresh_cropped = cv2.threshold(blur_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# cv2.imshow('result', thresh_cropped)
# cv2.waitKey(0)

# get the number of white pixels per line
for i in range(0, len(thresh_cropped)):
    pixels = np.sum(thresh_cropped[i])

    if pixels > 18000:
        # set the line to red
        for j in range(0, len(thresh_cropped[i])):
            thresh_cropped[i][j] = 0

# cv2.imshow('result', thresh_cropped)
# cv2.waitKey(0)

lines_tmp = cv2.HoughLinesP(thresh_cropped, rho=6, theta=np.pi/60, threshold=150, lines=np.array([]), minLineLength=20, maxLineGap=25)
# lines = lines_tmp
lines = []

# loop through all the lines and if the color behind the line is +- 10% of the color of white, then draw the line
percent = 0.5
for line in lines_tmp:
    for x1, y1, x2, y2 in line:
        color = lane_image[int((y1+y2)/2)][int((x1+x2)/2)]
        if (color[0] > 255*(1-percent) and color[1] > 255*(1-percent) and color[2] > 255*(1-percent)):
            print("white")
            lines.append(line)

print(len(lines))

# STEP 6: DRAW LINES ON THE IMAGE
def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0,255,0), thickness=3)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

image_with_lines = draw_the_lines(lane_image, lines)

# STEP 7: DISPLAY THE IMAGE
# cv2.imshow('result', image_with_lines)
# cv2.waitKey(0)
