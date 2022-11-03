# LANE DETECTION SYSTEM

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('lane.jpg')

# STEP 1: CONVERT TO GRAYSCALE
lane_image = np.copy(image)

gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

# STEP 2: APPLY GAUSSIAN BLUR
blur = cv2.GaussianBlur(gray, (5,5), 0)

# STEP 3: APPLY CANNY EDGE DETECTION
canny = cv2.Canny(blur, 50, 150)

# cv2.imshow('result', canny)
# cv2.waitKey(0)

blur_canny = cv2.GaussianBlur(canny, (5,5), 0)

thresh_blur = cv2.threshold(blur_canny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


width = thresh_blur.shape[1]
height = thresh_blur.shape[0]
manual_crop = [
    (0, height),
    (0, int(height/2)),
    (width, int(height/2)),
    (width, height)
]

# draw the manual crop region
cv2.line(lane_image, manual_crop[0], manual_crop[1], (255,0,0), 3)
cv2.line(lane_image, manual_crop[1], manual_crop[2], (255,0,0), 3)
cv2.line(lane_image, manual_crop[2], manual_crop[3], (255,0,0), 3)
cv2.line(lane_image, manual_crop[3], manual_crop[0], (255,0,0), 3)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# crop the thresh_blur image
cropped_image = region_of_interest(thresh_blur, np.array([manual_crop], np.int32))

# perspective transform the cropped image to get a top-down view
pts1 = np.float32([[0, height], [0, int(height/2)], [width, int(height/2)], [width, height]])
pts2 = np.float32([[0, height], [0, 0], [width, 0], [width, height]])
M = cv2.getPerspectiveTransform(pts1, pts2)
transformed_image = cv2.warpPerspective(cropped_image, M, (width, height))

cv2.imshow('result', transformed_image)
cv2.waitKey(0)

# generate a histogram of the transformed image
histogram = np.sum(transformed_image[transformed_image.shape[0]//2:,:], axis=0)

# display the histogram
# plt.plot(histogram)
# plt.show()

# get a list of pixels that are in each histogram spike
lanes = []

# when the histogram is greater than 0 set the new histogram value to 1
# otherwise set it to 0
hist_threshold = []
for i in range(len(histogram)):
    if histogram[i] > 0:
        hist_threshold.append(1)
    else:
        hist_threshold.append(0)

# plt.plot(hist_threshold)
# plt.show()

# find the start and end of each spike
start = 0
end = 0
for i in range(len(hist_threshold)):
    if len(lanes) >= 1:
        if lanes[-1][1] >= i:
            continue
    if hist_threshold[i] == 1:
        start = i
        for j in range(i, len(hist_threshold)):
            if hist_threshold[j] == 0 or j == len(hist_threshold) - 1:
                end = j
                break
        lanes.append((start, end))

# draw transparent lines on the image to show the lanes
for lane in lanes:
    cv2.line(lane_image, (lane[0], int(height/2)), (lane[0], height), (0,255,0), 3)
    cv2.line(lane_image, (lane[1], int(height/2)), (lane[1], height), (0,255,0), 3)

print(lanes)
# cv2.imshow('result', lane_image)
# cv2.waitKey(0)

# get the center of each lane
lane_centers = []
for lane in lanes:
    lane_centers_pixels = []
    # loop through each line in the transformed_image and find the center pixel that is white in the lane bounds
    for i in range(transformed_image.shape[0]):
        white_pixels = []
        for j in range(lane[0], lane[1]):
            if transformed_image[i][j] == 255:
                white_pixels.append(j)
        if len(white_pixels) > 0:
            lane_centers_pixels.append((i, int(sum(white_pixels)/len(white_pixels))))
    lane_centers.append(lane_centers_pixels)


# print(lane_centers)

# remove outliers from the lane centers
lane_centers_filtered = []
for lane in lane_centers:
    lane_centers_filtered.append([])
    for i in range(len(lane)):
        if i == 0:
            lane_centers_filtered[-1].append(lane[i])
        else:
            if abs(lane[i][1] - lane_centers_filtered[-1][-1][1]) < 50:
                lane_centers_filtered[-1].append(lane[i])

# draw the lane centers on the image
# for lane in lane_centers_inverted:
    # for point in lane:
        # cv2.circle(lane_image, (point[1], point[0]), 5, (0,0,127*lane_centers_inverted.index(lane)), -1)

# find lines that fit the lane centers
lane_lines = []
for lane in lane_centers_filtered:
    x = []
    y = []
    for point in lane:
        x.append(point[1])
        y.append(point[0])
    z = np.polyfit(y, x, 1)
    lane_lines.append(z)

# draw the lane lines on the image
for lane in lane_lines:
    for i in range(0, transformed_image.shape[0], 10):
        x = int(np.polyval(lane, i))
        cv2.circle(lane_image, (x, i), 2, (255,0,0), -1)

# cv2.imshow('result', lane_image)
# cv2.waitKey(0)
