import cv2
import numpy as np
import glob
import config
import pickle
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
        combined = np.zeros_like(l_binary_output)
        combined[(l_binary_output == 1) | (b_binary_output == 1)] = 1
        
        return combined
    
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