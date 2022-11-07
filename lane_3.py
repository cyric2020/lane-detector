# LANE DETECTION SYSTEM

import cv2
import numpy as np
import matplotlib.pyplot as plt
import image_cleaner

def start_convert(image):
    # STEP 1: CONVERT TO GRAYSCALE
    lane_image = np.copy(image)

    hls = cv2.cvtColor(lane_image, cv2.COLOR_RGB2HLS)

    sat_thres = 80
    sat = hls[:,:,2]
    sat_bin = np.zeros_like(sat)
    sat_bin[(sat > sat_thres)] = 1

    # plt.subplot(1, 2, 1)
    # plt.imshow(sat_bin, cmap='gray')

    sat_bin = image_cleaner.clean(sat_bin)

    # plt.subplot(1, 2, 2)
    # plt.imshow(sat_bin, cmap='gray')
    # plt.show()

    warped = image_cleaner.primary_crop(sat_bin)
    pers_mat = image_cleaner.get_pers_mat(sat_bin)

    # plt.imshow(warped, cmap='gray')
    # plt.show()

    def calculate_histogram(img):
        # calculate the histogram for the bottom half of the image
        bottom_half = img[img.shape[0]//2:,:]

        # calculate the histogram for the top half of the image
        top_half = img[:img.shape[0]//2,:]

        # combine the two halves
        histogram = np.sum(bottom_half, axis=0) + np.sum(top_half, axis=0)

        return histogram

    histogram = calculate_histogram(warped)

    # plot image and histogram side-by-side
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18,8))
    # ax1.imshow(warped, cmap='gray')
    # ax1.set_title('Warped Image')
    # ax2.plot(histogram)
    # ax2.set_title('Histogram')

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

    min_lane_gap = 100

    # if 2 spikes are within min_lane_gap pixels of each other, combine them
    i = 0
    while i < len(lanes) - 1:
        if i == len(lanes) - 1:
            break
        if lanes[i][1] + min_lane_gap >= lanes[i+1][0]:
            lanes[i] = (lanes[i][0], lanes[i+1][1])
            lanes.pop(i+1)
            i -= 1
        i += 1

    for lane in lanes:
        width = lane[1] - lane[0]
        area = 0
        for i in range(lane[0], lane[1]):
            area += histogram[i]
        # print(area)

        if area < 100:
            lanes.remove(lane)
    # print(lanes)

    # get the center of each lane
    lane_centers = []
    for lane in lanes:
        lane_centers_pixels = []
        # loop through each line in the transformed_image and find the center pixel that is white in the lane bounds
        for i in range(sat_bin.shape[0]):
            white_pixels = []
            for j in range(lane[0], lane[1]):
                if warped[i][j] == 1:
                    white_pixels.append(j)
            if len(white_pixels) > 0:
                # mid = int(sum(white_pixels)/len(white_pixels))
                mid = white_pixels[len(white_pixels)//2]
                # mid = white_pixels[0]
                # lane_centers_pixels.append((sat_bin.shape[0]-i, mid))
                lane_centers_pixels.append((i, mid))
        lane_centers.append(lane_centers_pixels)

    # find the line of best fit for each lane
    lane_lines = []
    for lane in lane_centers:
        x = []
        y = []
        for point in lane:
            x.append(point[0])
            y.append(point[1])
        lane_lines.append(np.polyfit(x, y, 2))

    # plot the lane lines on the image
    for lane in lane_lines:
        y = np.linspace(0, sat_bin.shape[0]-1, sat_bin.shape[0])

        x = np.polyval(lane, y)
        # plt.plot(x, y, color='red')

    # plt.imshow(warped, cmap='gray')
    # plt.show()

    # now overlay the lane lines on the original image
    # reverse the effect of the perspective warp on the points
    # then draw the lines on the original image
    all_points_lanes = []
    for lane in lane_lines:
        y = np.linspace(0, sat_bin.shape[0]-1, sat_bin.shape[0])

        x = np.polyval(lane, y)

        # reverse the perspective warp
        points = np.array([np.transpose(np.vstack([x, y]))])
        points = cv2.perspectiveTransform(points, np.linalg.inv(pers_mat))
        points = np.int32(points)

        # for point in points[0]:
            # all_points_lanes.append(point)
        all_points_lanes.append(points)

    all_points = []

    for a in range(len(all_points_lanes)):
        if a == 0:
            for b in all_points_lanes[a].tolist()[0]:
                all_points.append(b)
        else:
            for b in all_points_lanes[a].tolist()[0][::-1]:
                all_points.append(b)

    # print(all_points)

    cv2.fillPoly(lane_image, np.array([all_points], np.int32), (0, 255, 0))

    # make the fill 50% transparent
    lane_image = cv2.addWeighted(lane_image, 0.5, image, 0.5, 0)

    # draw the lines in blue
    for lane in all_points_lanes:
        cv2.polylines(lane_image, lane, False, (255, 0, 0), 2)


    rgb_lane_image = cv2.cvtColor(lane_image, cv2.COLOR_BGR2RGB)

    plt.imshow(rgb_lane_image)
    plt.show()

# image = cv2.imread('lane.jpg')
# start_convert(image)

plt.show()

cap = cv2.VideoCapture('inhigh.mp4')
i = 0
# while(cap.isOpened()):
while i < 1:
    ret, frame = cap.read()
    print(i)

    # height = 300
    height = 500
    height_diff = frame.shape[0]/height
    width = int(frame.shape[1]/height_diff)
    frame = cv2.resize(frame, (width, height))

    start_convert(frame)
    plt.pause(0.02)
    i+= 1