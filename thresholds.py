import cv2
import numpy as np
import glob
import config
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from calibrations import cameraCalibration

pickle_file_path = 'camera_cal/camera_distortion_pickle.p'

class thresholdedImage:
    def __init__(self, image):
        self.image = image

    # Thresholds derived from the knowledge article
    # https://knowledge.udacity.com/questions/32588
    # Apply the given thresholds to the l channel of the Luv color space
    def luv_l_thresh(self, img, thresh=(220, 255)):
        # Convert  the image to Luv color space
        luv = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
        luv_l = luv[:,:,0]
        # Scale
        luv_l = luv_l*(255/np.max(luv_l))
        # Apply thresholds
        binary_output = np.zeros_like(luv_l)
        binary_output[(luv_l > thresh[0]) & (luv_l <= thresh[1])] = 1
        return binary_output

    # Apply the given thresholds to the b channel of the Lab color space
    def lab_b_thresh(self, img,  thresh=(190,255)):
        # Convert  the image to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        lab_b = lab[:,:,2]
        # No scaling if there are no yellows in the image
        if np.max(lab_b) > 175:
            lab_b = lab_b*(255/np.max(lab_b))
        # 2) Apply a threshold
        binary_output = np.zeros_like(lab_b)
        binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
        return binary_output

    # Returns the binary image combining (OR) the binary thresholded
    # l channel from the Luv color space and b channel from Lab color
    # space
    def applyThresholds(self):

        l_binary_output = self.luv_l_thresh(self.image)
        b_binary_output = self.lab_b_thresh(self.image)

        # Combine Luv and Lab B color space channel thresholds
        self.result = np.zeros_like(l_binary_output)
        self.result[(l_binary_output == 1) | (b_binary_output == 1)] = 1

        # create the histogram for debug purposes
        self.histogram = self.getHistogram()
        return self.result

    #return the histogram of the thresholded image
    def getHistogram(self):
        binary = self.result
        self.histogram = np.zeros_like((self.result))
        self.histogram = np.dstack((self.histogram*255,self.histogram*255,self.histogram*255))
        # Grab only the bottom half of the image
        # Lane lines are likely to be mostly vertical nearest to the car
        bottom_half = binary[binary.shape[0]//2:,:]
        # Sum across image pixels vertically
        histogram_data = np.sum(bottom_half, axis=0)
        # Scale it for image size
        ysize = binary.shape[0]-1
        xsize = binary.shape[1]
        histogram_y = ysize - (np.interp(histogram_data, (histogram_data.min(), histogram_data.max()), (0, ysize)).astype(int))
        histogram_x = np.arange(xsize).astype(int)
        #prepare the point for polylines
        points = np.vstack((histogram_x, histogram_y)).T
        cv2.polylines(self.histogram, np.int32([points]), 0, (0,255,0), 5)
        return self.histogram

    #Alternate approach  suggested by reviewer
    #The performance of CLAHE+HSV is less than the current
    #approach in the project.
    def prepareThreshold_2():
        #preprocessing the images through a brightness adjustment filter
        #using a Contrast Limited Adaptive Histogram Equalization (CLAHE)
        #algorithm. This will help to better detect the yellow lane line road
        #markers that have over-cast shadows in the original image for
        #subsequent steps (e.g. getting a binary threshold color mask for
        #yellow and white lane lines).
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab_planes = cv2.split(lab)
        _clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        lab_planes[0] = _clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        self.image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        self.result = np.zeros((self.image_clahe.shape[0],self.image_clahe.shape[1]))

        hsv = cv2.cvtColor(self.image_clahe, cv2.COLOR_RGB2HSV)
        H = hsv[:,:,0]
        S = hsv[:,:,1]
        V = hsv[:,:,2]

        R = self.image_clahe[:,:,0]
        G = self.image_clahe[:,:,1]
        B = self.image_clahe[:,:,2]

        t_yellow_H = self.thresh(H,10,30)
        t_yellow_S = self.thresh(S,50,255)
        t_yellow_V = self.thresh(V,150,255)

        t_white_R = self.thresh(R,225,255)
        t_white_V = self.thresh(V,230,255)

        self.result[(t_yellow_H==1) & (t_yellow_S==1) & (t_yellow_V==1)] = 1
        self.result[(t_white_R==1)|(t_white_V==1)] = 1

    def thresh(self, image, thresh_min, thresh_max):
        ret = np.zeros_like(image)
        ret[(image >= thresh_min) & (image <= thresh_max)] = 1
        return ret
## The following functions are only for investigation purposes. This was used for the explorative
## to identify the best possible thresholds

pickle_file_path = 'camera_cal/camera_distortion_pickle.p'
output_path = 'output_images/'
test_images_path = 'test_images/*_warped.jpg'

# Code taken over from Lesson 7 - Gradients and Color Spaces: Applying Sobel
# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', sobel_kernel = 3, thresh = (0,255)):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# Code taken over from Lesson 7 - Gradients and Color Spaces: Magnitude of Gradient
# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Code taken over from Lesson 7 - Gradients and Color Spaces: Direction of Gradient
# Define a function to threshold an image for a given range and Sobel kernel for directionsobel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output.astype(np.uint8)

# Iterates the warped images in the test_images folder and prints the various
# outputs from Sobel gradients and S-binary
def print_gradient_thresholds():
    cc = cameraCalibration(pickle_file_path)
    # iterate through all the images in the test images folder
    images = glob.glob(test_images_path)

    f, axs = plt.subplots(len(images) , 6, figsize=(40, 40), sharex = True, sharey=True)
    axs = axs.ravel()

    for idx, fname in enumerate(images):
        image = mpimg.imread(fname)
        # undistort the test image
        ret, undistorted = cc.undistort(image)

        gray = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
        # Apply each of the thresholding functions
        #Convert to HSL
        hls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS).astype(np.float)
        #read saturation channel
        s = hls[:,:,2].astype(np.uint8)

        gradx = abs_sobel_thresh(gray, orient='x',
                                 sobel_kernel = config.kernel_size,
                                 thresh = config.sobel_x_thresholds)

        grady = abs_sobel_thresh(gray, orient = 'y',
                                 sobel_kernel = config.kernel_size,
                                 thresh=config.sobel_y_thresholds)

        mag_binary = mag_thresh(gray, sobel_kernel = config.kernel_size,
                                thresh = config.magnitude_thresholds)

        dir_binary = dir_threshold(gray, sobel_kernel = config.kernel_size,
                                   thresh = config.direction_thresholds)

        s_binary = np.zeros_like(s).astype(np.uint8)
        s_binary[(s > config.s_binary_thresholds[0]) & (s <= config.s_binary_thresholds[1])] = 1

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | (mag_binary == 1) | (s_binary ==1) ] = 1

        print ('Working on file: ' + fname + '...')
        axs[(idx*6)].imshow(image)
        if idx == 0:
            axs[(idx*6)].set_title('Original Image', fontsize=30)

        axs[(idx*6)+1].imshow(gradx, cmap='gray')
        if idx == 0:
            axs[(idx*6)+1].set_title('sobelx'+ str(config.sobel_x_thresholds), fontsize=30)

        axs[(idx*6)+2].imshow(grady, cmap='gray')
        if idx == 0:
            axs[(idx*6)+2].set_title('sobely' + str(config.sobel_y_thresholds), fontsize=30)

        axs[(idx*6)+3].imshow(mag_binary, cmap='gray')
        if idx == 0:
            axs[(idx*6)+3].set_title('magnitude' + str(config.magnitude_thresholds), fontsize=30)

        axs[(idx*6)+4].imshow(s_binary, cmap='gray')
        if idx == 0:
            axs[(idx*6)+4].set_title('s-binary' + str(config.s_binary_thresholds), fontsize=30)

        axs[(idx*6)+5].imshow(combined, cmap='gray')
        if idx == 0:
            axs[(idx*6)+5].set_title('combined((x&y)|mag|s)' , fontsize=30)

        f.tight_layout()

    plt.savefig(output_path + 'results_gradient_thresholds.jpg')
    print ('Results saved at ' + output_path + 'results_gradient_thresholds.jpg')

def luv_l_thresh( img, thresh=(220, 255)):
    # Convert  the image to Luv color space
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    luv_l = luv[:,:,0]
    # Scale
    luv_l = luv_l*(255/np.max(luv_l))
    # Apply thresholds
    binary_output = np.zeros_like(luv_l)
    binary_output[(luv_l > thresh[0]) & (luv_l <= thresh[1])] = 1
    return binary_output

def lab_b_thresh( img,  thresh=(190,255)):
    # Convert  the image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = lab[:,:,2]
    # No scaling if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    return binary_output

# Iterates the warped images in the test_images folder and prints the
# outputs from Luv L and Lab B channel with thresholds
def print_color_thresholds():
    cc = cameraCalibration(pickle_file_path)
    # iterate through all the images in the test images folder
    images = glob.glob(test_images_path)

    f, axs = plt.subplots(len(images) , 4, figsize=(40, 40), sharex = True, sharey=True)
    axs = axs.ravel()

    for idx, fname in enumerate(images):
        image = mpimg.imread(fname)
        ret, undistorted = cc.undistort(image)

        l_binary_output = luv_l_thresh(image)
        b_binary_output = lab_b_thresh(image)

        # Combine Luv and Lab B color space channel thresholds
        combined = np.zeros_like(l_binary_output)
        combined[(l_binary_output == 1) | (b_binary_output == 1)] = 1

        print ('Working on file: ' + fname + '...')
        axs[(idx*4)].imshow(undistorted)
        if idx == 0:
            axs[(idx*4)].set_title('Undistorted image', fontsize=30)

        axs[(idx*4)+1].imshow(l_binary_output, cmap='gray')
        if idx == 0:
            axs[(idx*4)+1].set_title('l(Luv) - White line' +  str(config.l_binary_thresholds), fontsize=30)

        axs[(idx*4)+2].imshow(b_binary_output, cmap='gray')
        if idx == 0:
            axs[(idx*4)+2].set_title('b(Lab) - Yellow line'+ str(config.b_binary_thresholds), fontsize=30)

        axs[(idx*4)+3].imshow(combined, cmap='gray')
        if idx == 0:
            axs[(idx*4)+3].set_title('l|b', fontsize=30)

        f.tight_layout()
    plt.savefig(output_path + 'results_color_thresholds.jpg')
    print ('Results saved at ' + output_path + 'results_color_thresholds.jpg')

def main():
    print_color_thresholds()
    print_gradient_thresholds()

if __name__ == '__main__':
    main()
