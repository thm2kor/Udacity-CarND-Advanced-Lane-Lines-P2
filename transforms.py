import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import os
#project specific modules
import config
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

# define utility function for selecting roi
def selectRoi(im):
    # mask out any pixels outside of the roi
    bottomLeft = [0,config.IMAGE_HEIGHT*0.95]
    topLeft = [config.IMAGE_WIDTH*0.10, config.IMAGE_HEIGHT*0.65]
    topRight = [config.IMAGE_WIDTH*0.95, config.IMAGE_HEIGHT*0.65]
    bottomRight = [config.IMAGE_WIDTH, config.IMAGE_HEIGHT*0.95]

    vertices = np.array([[bottomLeft,topLeft,topRight,bottomRight]],dtype=np.int32)
    mask = np.zeros_like(im)
    cv2.fillPoly(mask,vertices,255)
    return cv2.bitwise_and(im,mask)

# define utility function for determining line intersection

def findIntersection(lines):
    # find the point minimizing lsq distance from all lines
    # if the lines all intersect, this is the intersection point
    numLines = len(lines)
    a = np.zeros((numLines,2),dtype=np.float32)
    b = np.zeros((numLines,),dtype=np.float32)
    for n,line in enumerate(lines):
        for x1,y1,x2,y2 in line:
            slope = (y2-y1) / float(x2-x1)
            a[n] = np.array([slope,-1],dtype=np.float32)
            b[n] = slope * x1 - y1     # this is -1 times the intercept
    return np.linalg.lstsq(a,b,rcond=None)[0]

# function for finding vanishing point in a given image (prefer)
def findVanishingPoint(file, mtx, dst):
    # determine vanishing point in image
    # analyze straight-road images used to find vanishing point
    image = cv2.imread(file)
    image = cv2.undistort(image,mtx,dst)
    # smooth with gaussian blur
    smoothed = cv2.GaussianBlur(image,(3,3),0)
    # find edges with canny
    edges = cv2.Canny(smoothed,50,400)
    # apply roi mask
    edgesRoi = selectRoi(edges)
    #Parameters for Hough transform
    HOUGH_DIST_RES = 0.5                # hough line finder distance resolution (pixels)
    HOUGH_ANGLE_RES = 3.14159/180       # hough line finder angle resolution (rads)
    HOUGH_THRESHOLD = 20                # hough line finder threshold
    HOUGH_MIN_LINE = 60                 # hough line finder min line length (pixels)
    HOUGH_MAX_GAP = 120                 # hough line finder max line gap (pixels)
    #find lines with hough; use probabilistic version to increase speed
    lines = cv2.HoughLinesP(edgesRoi, HOUGH_DIST_RES, HOUGH_ANGLE_RES,
                                HOUGH_THRESHOLD, None,
                                HOUGH_MIN_LINE, HOUGH_MAX_GAP)
    #identify best overlap point of lines - this is the vanishing point
    vp = findIntersection(lines)
    # save image
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1,y1), (x2,y2), (0,255,0), thickness=2)
    cv2.circle(image,(vp[0],vp[1]),8,(0,255,0),2)
    savepath = os.path.join(output_path,os.path.basename(file)[:-4] + '_vp.jpg')
    cv2.imwrite(savepath,image)
    print ('Result with vanishing point saved at ' + savepath)
    return vp

def deriveSrcDestRects(filename):
    cc = cameraCalibration(pickle_file_path)
    vp = findVanishingPoint(filename, cc.mtx, cc.dist)
    print  ('Vanishing point based on ' + filename +  ' is ' + str(vp))

    xVP,yVP = int(vp[0]),int(vp[1])
    xBottomLeft = int(0.10 * config.IMAGE_WIDTH)
    xBottomRight = int(0.95 * config.IMAGE_WIDTH)
    yTop = int(config.IMAGE_HEIGHT * 0.65)
    yBottom = int(config.IMAGE_HEIGHT * 0.95)

    # calculate x positions of upper ROI corners
    # x = my + b; m = (x1-x0)/(y1-y0); b = x0 - m * y0
    leftM = (xVP - xBottomLeft) / float(yVP - yBottom)
    leftB = xBottomLeft - leftM * yBottom
    rightM = (xVP - xBottomRight) / float(yVP - yBottom)
    rightB = xBottomRight - rightM * yBottom
    xTopLeft = int(leftM * yTop + leftB)
    xTopRight = int(rightM * yTop + rightB)

    src = [(xTopLeft,yTop),(xBottomLeft,yBottom),(xBottomRight,yBottom),(xTopRight,yTop)]
    dest = [(0,0),(0,config.IMAGE_HEIGHT),(config.IMAGE_WIDTH,config.IMAGE_HEIGHT),(config.IMAGE_WIDTH,0)]
    print  ('src points are ' + str(src))
    print  ('dest points are ' + str(dest))

def main():
    parser = argparse.ArgumentParser('perspective transformation module')

    parser.add_argument('--warp', dest='warp', help='unwarps the given image and stores in the result in the output_images folder')
    parser.add_argument('--unwarp', dest='unwarp', help='unwarps the given image and stores in the result in the output_images folder')
    parser.add_argument('--comp', dest='no_compare', action='store_true', help='when set, the output will be a side-by-side comparison of original and warped image')
    parser.add_argument('--findvp', dest='find_vp', help='the given image will be used to define the src and dest rects based on the vanishing point method')
    args = parser.parse_args()

    if args.warp is not None:
        warp(args.warp, args.no_compare)

    if args.unwarp is not None:
        unwarp(args.unwarp,args.no_compare)

    if args.find_vp is not None:
        deriveSrcDestRects(args.find_vp)

if __name__ == '__main__':
    main()
