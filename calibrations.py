# This module computes the camera calibration matrix and 
# distortion coefficients given a set of 20 chessboard images.
# The chessboard images are available at ./camera_cal
# The computed coefficients and camera matrix are archived in a 
# 
import cv2
import glob
import pickle
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt      
import matplotlib.image as mpimg

import config

##TODO: Adapt the below paths incase the folder structure of the project changes
pickle_file_path = 'camera_cal/camera_distortion_pickle.p'
calibration_file_path = 'camera_cal/calibration*.jpg'
output_path = 'output_images/'
test_images_path = 'camera_cal/calibration*.jpg'

class cameraCalibration: 
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    mtx = [] # matrox coefficients
    dist = [] # distortio coefficients
    status = False
    def __init__(self, pickle_file_path):
        try:
            # Read in the saved objpoints and imgpoints
            dist_pickle = pickle.load( open( pickle_file_path, 'rb' ) )
            #print ('loading calibration data from ' + pickle_file_path )
            self.objpoints = dist_pickle['objpoints']
            self.imgpoints = dist_pickle['imgpoints']
            self.mtx = dist_pickle['mtx']
            self.dist = dist_pickle['dist']
            self.status = True
        except:
            print ("calibration file not available.")
            
    def undistort(self, image):
        if self.status == True:
            undist = cv2.undistort(image, self.mtx, self.dist, None, None)
            return True, undist
        else:
            return False, None
        
## one time configuration to calibrate the camera

## determine camera calibration parameter and save them to a pickle file 

# This function is a simple wrapper around cv.findChessboardCorners and cv2.cv.findChessboardCorners. Some parts of the code are taken over the course material.
# It attempts to iterat through the training images (calibration*.jpg") files located in the "camera_cal" folder.
# To support visual inspection, the identified corners are rendered directly # on the jpg file and saved in the "output_images" folder with the postfix "_corners".
# The results of the camera calibration ( calibration matrix,, distortion co-efficients, the 3D obj points and 2D image points are saved on a pickle file
# NOTE: Atleast 3 calibration jpg failed to detect the edges. Data from these images are not considered for the calibration.
def save_camera_calib_data():
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # prepare object points
    objp = np.zeros((config.nx*config.ny, 3), np.float32)
    #no changes to the z-axis
    objp[:,:2] = np.mgrid[0:config.nx, 0:config.ny].T.reshape(-1,2)
    
    # iterate through all the images in the camera cal folder
    print ('Searching for calibration files in ' + calibration_file_path + '...')
    images = glob.glob(calibration_file_path)
    for idx, fname in enumerate(images):
        
        img = cv2.imread(fname)
        copy = np.copy(img)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (config.nx,config.ny), None)
    
        # If found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
            #objp will be the same for all detected images since
            #they represent a real chessboard
            objpoints.append(objp)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (config.nx,config.ny), corners, ret)
            
            
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(copy)
            ax1.set_title('Original Image', fontsize=30)
            ax2.imshow(img)
            ax2.set_title('Indentified Corners', fontsize=30)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.savefig(output_path + os.path.basename(fname)[:-4] + '_corners.jpg')
            # save the image with corners (for visual inspection)
            print ('Corners found in ' + os.path.basename(fname) + 
                   '.Result file with detected corners saved at ' + \
                   output_path + os.path.basename(fname)[:-4] + '_corners.jpg')

        else:
            print ('Unable to find corners in ' + os.path.basename(fname))
    
    if len(images):
        # Camera calibration based on the  given object points, image points, 
        # and the shape of the grayscale image:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle['mtx'] = mtx
        dist_pickle['dist'] = dist
        dist_pickle['objpoints'] = objpoints
        dist_pickle['imgpoints'] = imgpoints
        try:
            pickle.dump( dist_pickle, open( pickle_file_path, 'wb' ) )
            print ('####### Calibration data are saved at ' + pickle_file_path + ' #######')
        except:
            print ('####### Unable to save camera calibration data. #######')

## verify the calibration data using visual inspection method
# This function unpickles the calibration data available at the camera_cal folder to generate the undistored image.
# To support visual inspection, the side-by-side image of the calibration*.jpg with its corresponding undistored image is saved in the "output_images" folder with the postfix "_undistorted"
# If the function fails due to missing pickle file, please run the function save_camera_calib_data. 
def verify_calib_data():
    mtx = []
    dst =[]
    # iterate through all the images in the camera cal folder
    try:
        # Read in the saved objpoints and imgpoints
        dist_pickle = pickle.load( open( pickle_file_path, 'rb' ) )
        print ('loading calibration data from ' + pickle_file_path )
        mtx = dist_pickle['mtx']
        dist = dist_pickle['dist']
        
    except:
        print ("calibration file not available. Run python calibrations.py --refresh")
        exit()
        
    print ('Searching for images in ' + test_images_path + '...')
    images = glob.glob(test_images_path)
    for idx, fname in enumerate(images):    
        # Read in an image
        image = mpimg.imread(fname)
        undistorted = cv2.undistort(image, mtx, dist, None, None) #cal_undistort(image, objpoints, imgpoints)
        # save the undistorted image in the output_image folder
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(undistorted)
        ax2.set_title('Undistored Image', fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(output_path + os.path.basename(fname)[:-4] + '_undistored.jpg')
        print ('Comparison image saved at ' + fname[:-4] + '_undistored.jpg')

# based on the previously calibrated camera and distortion coefficients cached 
# in the <pickle_file_path> undistort the given image
def processImage(filename):
    cc = cameraCalibration(pickle_file_path)
    image = mpimg.imread(filename)
    if image is not None:
        ret, undist = cc.undistort(image)
        if ret:
            # save the undistorted image in the output_image folder
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(image)
            ax1.set_title('Original Image', fontsize=30)
            ax2.imshow(undist)
            ax2.set_title('Undistored Image', fontsize=30)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            result_file = output_path + os.path.basename(filename)[:-4] + '_undistored.jpg'
            plt.savefig(result_file)
            print ('Comparison image saved at ' + result_file)
                
# Main function
def main():
    # from a conda environment, call 'python camera_cal --refresh' if the calibration needs to 
    # be recreated.
    parser = argparse.ArgumentParser(description='Flag to do fresh camera calibration')
    
    parser.add_argument('--corners', dest='corners', action='store_true', help='identifies corners in ALL the *.jpg file in camera_cal folder')
    parser.add_argument('--undistort', dest='undistort', action='store_true', help='undistorts ALL *.jpg file in camera_cal folder')
    parser.add_argument('--filename', dest='filename', help='undistort the given image and stores in the result in the output_images folder')
    
    args = parser.parse_args()
    ## read-in the chessbaord images and cache the camera matrix and distortion
    ## coeefficients in a pickle file
    if args.corners == True:
        save_camera_calib_data()
    
    ## reads ALL the calibration images and undistorts the images. A comparative 
    ## image is saved in the output folder
    if args.undistort == True:
        verify_calib_data()
    
    ## process single file
    if args.filename is not None:
        processImage(args.filename)
        
if __name__ == '__main__':
    main()