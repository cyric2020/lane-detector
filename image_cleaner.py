import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

def average_white(image):
    # draw the mean position of the pixels
    meanx = np.mean(np.where(image == 1)[1])
    meany = np.mean(np.where(image == 1)[0])
    cv2.circle(image, (int(meanx), int(meany)), 5, (255, 255, 255), -1)

    return image

def get_perspective(image):

    # use the config from lane_3
    
    return [
        (0, image.shape[0]),
        (int(2*image.shape[1]/5), int(image.shape[0]/2 + image.shape[0]/int(config['CLEANING']['Mid_Redux']))),
        (int(image.shape[1]-2*image.shape[1]/5), int(image.shape[0]/2 + image.shape[0]/int(config['CLEANING']['Mid_Redux']))),
        (image.shape[1], image.shape[0])
    ]

def get_pers_mat(image):
    perspective = get_perspective(image)
    # perspective warp the sat_bin image
    pers_mat = cv2.getPerspectiveTransform(np.float32(perspective), np.float32([[0, image.shape[0]], [0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]]]))
    return pers_mat

def primary_crop(image):
    # crop the image to the perspective
    pers_mat = get_pers_mat(image)
    warped = cv2.warpPerspective(image, pers_mat, (image.shape[1], image.shape[0]))

    return warped

def angle_clean(image, incriment, draw=False):
    lane_origin = (image.shape[1]//2, image.shape[0])

    ray_image = np.zeros_like(image)

    angle = 0
    distances = []
    angles = []
    while angle < 180:
        # get the first white pixel in the image closest to the origin

        max_x = int(lane_origin[0] + (image.shape[1]//2) * np.cos(np.deg2rad(angle)))
        max_y = int(lane_origin[1] - (image.shape[0]//2) * np.sin(np.deg2rad(angle)))

        line_length = int(np.sqrt((max_x - lane_origin[0])**2 + (max_y - lane_origin[1])**2))

        for i in range(line_length):
            x = int(lane_origin[0] + i * np.cos(np.deg2rad(angle)))
            y = int(lane_origin[1] - i * np.sin(np.deg2rad(angle)))

            if image[y-1, x] == 1:
                ray_image[y-1, x] = 1
                if draw == True:
                    cv2.line(ray_image, lane_origin, (x, y-1), (255, 0, 0), 1)
                distances.append(i)
                break

        angles.append(angle)
        if len(distances) < 1:
            distances.append(0)
        if len(distances) < len(angles):
            # append the previous distance
            distances.append(distances[-1])

        # increment the angle
        angle += float(config['CLEANING']['Angle_Increment'])
    
    return distances, angles, ray_image

def center_clean(image, incriment, draw=False):
    lane_origin = (image.shape[1]//2, image.shape[0]-image.shape[0]//4)

    ray_image = np.zeros_like(image)

    angle = 0
    distances = []
    angles = []
    while angle < 360:
        # get the first white pixel in the image closest to the origin

        max_x = int(lane_origin[0] + (image.shape[1]//2) * np.cos(np.deg2rad(angle)))
        max_y = int(lane_origin[1] - (image.shape[0]//2) * np.sin(np.deg2rad(angle)))

        line_length = int(np.sqrt((max_x - lane_origin[0])**2 + (max_y - lane_origin[1])**2))

        for i in range(line_length):
            x = int(lane_origin[0] + i * np.cos(np.deg2rad(angle)))
            y = int(lane_origin[1] - i * np.sin(np.deg2rad(angle)))

            try:
                if image[y, x] == 1:
                    ray_image[y, x] = 1
                    if draw == True:
                        cv2.line(ray_image, lane_origin, (x, y), (255, 0, 0), 1)
                    distances.append(i)
                    break
            except:
                pass

        angles.append(angle)
        if len(distances) < 1:
            distances.append(0)
        if len(distances) < len(angles):
            # append the previous distance
            distances.append(distances[-1])

        # increment the angle
        angle += float(config['CLEANING']['Angle_Increment'])
    
    return distances, angles, ray_image

def clean(image):

    # get pixel clump sizes
    pixel_clumps = cv2.connectedComponentsWithStats(image, 8, cv2.CV_32S)

    # if the number of pixels in a clump is less than 5 then remove it
    min_thresh = 15
    max_thresh = 20000
    for i in range(1, pixel_clumps[0]):
        if pixel_clumps[2][i][4] < min_thresh:
            image[pixel_clumps[1] == i] = 0
        elif pixel_clumps[2][i][4] > max_thresh:
            image[pixel_clumps[1] == i] = 0

    distances, angles, ray_image = angle_clean(image, 0.1, False)

    if config['CLEANING']['Run_Center_Rays'] == 'True':
        distances, angles, ray_image = center_clean(ray_image, 0.1, False)

    return ray_image

    # avg_x = np.mean(np.where(ray_image == 1)[1])
    # avg_y = np.mean(np.where(ray_image == 1)[0])

    # ray_image[int(avg_y), int(avg_x)] = 1

    # angles = angles[::-1]


    # TODO: Might impliment later
    # plot the gradient of the distances
    # distance_gradient = np.gradient(distances)

    # set all distance gradients below 10 to 0
    # distance_gradient[np.abs(distance_gradient) < 10] = 0
    # distance_gradient[np.abs(distance_gradient) > 0] = 1

    # plot the distances against the angles
    # plt.subplot(2, 1, 1)
    # plt.imshow(ray_image, cmap='gray')
    # plt.subplot(2, 1, 2)
    # plt.plot(angles, distances)
    # plot angles in reverse
    # plt.plot(angles, distance_gradient)
    # plt.show()
        

    warped_image = primary_crop(ray_image)

    distances_warped, angles_warped, warped_image_angle_cleaned = angle_clean(warped_image, 0.5, draw=True)

    # plt.subplot(1, 2, 1)
    # plt.imshow(warped_image, cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(warped_image_angle_cleaned, cmap='gray')
    # plt.show()

    # show the histogram
    # plt.plot(histogram)
    # plt.show()

    # show both images
    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(ray_image, cmap='gray')
    # plt.show()

    # save ray_image to the output folder
    # convert ray_image to a format that can be saved
    ray_image = np.uint8(ray_image * 255)
    # save the image with the date in the name
    # cv2.imwrite('output/ray_image_{}.png'.format(
        # datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), ray_image)

    # save warped_image to the output folder
    # convert warped_image to a format that can be saved
    warped_image = np.uint8(warped_image * 255)
    # save the image with the date in the name
    # cv2.imwrite('output/warped_image_{}.png'.format(
        # datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), warped_image)

    # return ray_image

    # plt.imshow(image, cmap='gray')
    # plt.show()

# clean(sat_bin)
# plt.show()

# loop through every frame in in.mp4
# cap = cv2.VideoCapture('inhigh.mp4')
# i = 0
# # while(cap.isOpened()):
# while i < 1:
#     ret, frame = cap.read()
#     print(i)

#     # convert frame to 300 pixels high (maintain aspect ratio) use cv2
#     height = 300
#     height_diff = frame.shape[0]/height
#     width = int(frame.shape[1]/height_diff)
#     frame = cv2.resize(frame, (width, height))

#     hls_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)

#     sat_frame = hls_frame[:,:,2]
#     sat_bin_frame = np.zeros_like(sat_frame)
#     sat_bin_frame[(sat_frame > 80)] = 1

#     clean(sat_bin_frame)
#     plt.pause(0.01)
#     i+= 1

    