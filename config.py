# This module holds all the configuration parameters
#
# Parameters for camera calibration
debug_mode = True   # In this model, additional windows with intermediate stages are displayed
                    # Relevant only for images
                    # can be overridden with the flag --debug 
                    # python main.py <path_to_image> --debug

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
nx = 9                                  # the number of inside corners in x
ny = 6                                  # the number of inside corners in y

kernel_size = 15                        # sobel kernel size
sobel_x_thresholds = (20, 100)          # sobel thresholds for gradients in 'x' direction
sobel_y_thresholds = (20, 100)          # sobel thresholds for gradients in 'y' direction

magnitude_thresholds = (30, 170)        # overall magnitude of the gradients magnitude is
                                        # the square root of the sum of squares of 
                                        # gradients in 'x' and 'y' direction 

direction_thresholds = (0.7, 1.3)       # thresholds for detecting edges in a particular
                                        # direction or orientation

b_binary_thresholds = (190, 255)        # thresholds for b channel thresholds in Lab color space
s_binary_thresholds = (110, 255)        # thresholds for s channel thresholds in HSV color space
l_binary_thresholds = (220, 225)        # thresholds for l channel thresholds in Luv color space
v_binary_thresholds = (230, 255)        # thresholds for v channel thresholds in HSV color space
