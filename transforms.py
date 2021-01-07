import cv2
import numpy as np
import matplotlib.pyplot as plt      
import matplotlib.image as mpimg
import argparse
import os
#project specific modules
from calibrations import cameraCalibration

pickle_file_path = 'camera_cal/camera_distortion_pickle.p'
calibration_file_path = 'camera_cal/calibration*.jpg'
output_path = 'output_images/'
test_images_path = 'camera_cal/calibration*.jpg'

class perspectiveTransform: 
    M = []
    Minv = []
    def __init__(self):
        self.src = np.float32([(575,465),   # Top Left
                          (710,465),        # Top Right
                          (1050,680),       # Bototm Right
                          (260,680)])       # Bottom Left
        self.dest = np.float32([(450,0),    # Top Left
                          (830,0),          # Top Right
                          (830,720),        # Bottom Right
                          (450,720)])       # Bottom Left
        ##TODO : Adapt the value based on the following article
        # https://knowledge.udacity.com/questions/22331
        self.M = cv2.getPerspectiveTransform(self.src,self.dest)
        self.Minv = cv2.getPerspectiveTransform(self.dest,self.src)
    
    def warp (self, image):
        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    
    def unwarp (self, image):
        return cv2.warpPerspective(image, self.Minv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    
    
def warp(filename, comp = False):
    image = mpimg.imread(filename)
    #copy = np.copy(image)
    if image is not None:
        #load camera calibration data
        cc = cameraCalibration(pickle_file_path)    
        #Undistort the image
        ret, undistorted = cc.undistort(image )
        
        if ret == False:
            print ('Distortion correction failed. Tranform not done')
            return
        
        pt = perspectiveTransform()
        warped = pt.warp(undistorted)
        
        #uncomment the below code for debugging purposes
        #cv2.fillPoly(image, np.int_([pt.src]), (0,255, 0))
        #result_file = output_path + os.path.basename(filename)[:-4] + '.jpg'
        #mpimg.imsave(result_file, warped)
        
        if comp:
            # save the transform image in the output_image folder
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(image)
            ax1.set_title('Original Image (Undistorted)', fontsize=30)
            ax2.imshow(warped)
            ax2.set_title('Warped Image', fontsize=30)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            result_file = output_path + os.path.basename(filename)[:-4] + '_warped_cmp.jpg'
            plt.savefig(result_file)
            print ('Comparison image saved at ' + result_file)
        else:
            result_file = output_path + os.path.basename(filename)[:-4] + '_warped.jpg'
            mpimg.imsave(result_file, warped) 
            print ('Warped image saved at ' + result_file)


def unwarp(filename, comp=False):
    print (filename)
    image = mpimg.imread(filename)
    if image is not None:
        pt = perspectiveTransform()
        unwarped = pt.unwarp(image)
        if comp:
            # save the transform image in the output_image folder
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(image)
            ax1.set_title('Warped Image', fontsize=30)
            ax2.imshow(unwarped)
            ax2.set_title('Unwarped Image', fontsize=30)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            result_file = output_path + os.path.basename(filename)[:-4] + '_unwarped_cmp.jpg'
            plt.savefig(result_file)
            print ('Comparison image saved at ' + result_file)
        else:
            result_file = output_path + os.path.basename(filename)[:-4] + '_unwarped.jpg'
            mpimg.imsave(result_file, unwarped)          
            print ('Warped image saved at ' + result_file)

def main(): 
    parser = argparse.ArgumentParser('perspective transformation module')
    
    parser.add_argument('--warp', dest='warp', help='unwarps the given image and stores in the result in the output_images folder')
    parser.add_argument('--unwarp', dest='unwarp', help='unwarps the given image and stores in the result in the output_images folder')
    parser.add_argument('--comp', dest='no_compare', action='store_true', help='when set, the output will be a side-by-side comparison of original and warped image')
    args = parser.parse_args()
    
    if args.warp is not None:
        warp(args.warp, args.no_compare)
    
    if args.unwarp is not None:
        unwarp(args.unwarp,args.no_compare)
        
if __name__ == '__main__':
    main()