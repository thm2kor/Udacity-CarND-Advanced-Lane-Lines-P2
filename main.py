import matplotlib.image as mpimg
import cv2
import argparse
import config
import os
import numpy as np
import imageio
from moviepy.editor import VideoFileClip

#project specific modules
from thresholds import thresholdedImage
from transforms import perspectiveTransform
from lanes import drivingLane
from calibrations import cameraCalibration

pickle_file_path = 'camera_cal/camera_distortion_pickle.p'
output_path = 'output_images/'
output_path_video = 'output_videos/'
test_images_path = 'test_images/*.jpg'

#class - the default function serves as the pipeline for the images and video frames
class pipeline:
    def __init__(self):
        # Load the calibration parameters from the pickle file
        self.cc = cameraCalibration(pickle_file_path)
        # track object encapsulates the lane lines detection and the
        # calculation of the intercepts, lane width and vehicle pos
        self.track = drivingLane()
        # pt encapsulates the transform matrices, warp and unwarp function
        self.pt = perspectiveTransform()

    #Pipeline for Lane processing
    def __call__(self, image, mode=0):
        distorted = np.copy(image)
        #Undistort the given image
        ret, undistorted = self.cc.undistort(distorted)
        #apply perspective transform to the undistorted image
        self.warped = self.pt.warp(undistorted)
        #color threshold the frames to filter the lane lines
        binary = thresholdedImage(self.warped)
        self.edges = binary.applyThresholds()
        #find the lines based on the detected edges
        self.track.detect_lines(self.edges)
        #overlay the tracks on the distorted image
        filled_track = self.track.overlay_lanes(distorted, self.edges)
        #unwarp the combined image
        if config.debug_mode == True and mode == 0:#1= video mode
            result = self.show_debug_info(self.edges)
            #in the debug mode, only return edges and the line fits
            #TODO: Update the texts in white fonts
            return result
        else:
            unwarp = self.pt.unwarp(filled_track)
            #Combine the result with the original image
            result = cv2.addWeighted(distorted, 1, unwarp, 0.5, 0)
            #display the curve radius and position
            result = self.add_header(result)
            #return result
            return result

    #Adds the calculated radius of curvature and vehicle position on the video frames
    def add_header(self, image):
        result = np.copy(image)
        font = cv2.FONT_HERSHEY_SIMPLEX

        radius = self.track.get_curve_radius()
        position = self.track.get_vehicle_pos(image.shape)
        if radius > 2000:
            text = 'Radius of curvature: (NA) Straight line'
        else:
            text = 'Radius of curvature: ' + '{:05.2f}'.format(radius) + ' m'
        cv2.putText(result, text, (50,70), font, 1.5, (255,255,255), 2, cv2.LINE_AA)

        #text = 'Radius of curvature: L ' + '{:05.2f}'.format(self.track.leftline.radius_of_curvature) + 'm'
        #cv2.putText(result, text, (50, 120), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
        #text = 'Radius of curvature: R ' + '{:05.2f}'.format(self.track.rightline.radius_of_curvature) + 'm'
        #cv2.putText(result, text, (50, 170), font, 1.0, (255,255,255), 2, cv2.LINE_AA)

        if position > 0:
            direction = 'right of center'
        else:
            direction = 'left of center'
        text = 'Vehicle is {:04.2f}'.format(abs(position)) + 'm ' + direction
        cv2.putText(result, text, (50,120), font, 1.5, (255,255,255), 2, cv2.LINE_AA)

        return result

    ##Functions which draws the left and right line fits on the edges
    def show_debug_info(self, edges):
        debug_bin = np.dstack((edges*255, edges*255, edges*255))

        # overhead with all fits added (bottom right)
        debug_bin_points = np.copy(debug_bin)

        debug_bin_points[self.track.leftline.ally, self.track.leftline.allx] = [255, 0, 0]
        debug_bin_points[self.track.rightline.ally, self.track.rightline.allx] = [0, 0, 255]

        for i, fit in enumerate(self.track.leftline.current_fit):
            debug_bin_points = self.draw_lines_on_image(debug_bin_points, fit, (155,0, 0), 3)
        for i, fit in enumerate(self.track.rightline.current_fit):
            debug_bin_points = self.draw_lines_on_image(debug_bin_points, fit, (0, 0, 155), 3)

        debug_bin_points = self.draw_lines_on_image(debug_bin_points, self.track.leftline.best_fit, (0,255,0))
        debug_bin_points = self.draw_lines_on_image(debug_bin_points, self.track.rightline.best_fit, (0,255,0))

        return debug_bin_points

    #Helper function for debug info - draws a given line on to an image
    def draw_lines_on_image(self, image, fit, plot_color, thickness=5):
        if fit is None:
            return image
        result = np.copy(image)
        h = result.shape[0]
        ploty = np.linspace(0, h-1, h)
        plotx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        pts = np.array([np.transpose(np.vstack([plotx, ploty]))])
        cv2.polylines(result, np.int32([pts]), isClosed=False, color=plot_color, thickness=8)
        return result

# function which reads in a video and prepares the frames for the pipeline
def processVideo(filename, start, end):
    clip = VideoFileClip(filename)
    if start > 0 and end > 0:
        clip = clip.subclip(start,end)
    snapshot = clip.fl_image(pipeline() )

    if config.debug_mode == False:
        result_file_name = 'result_' + os.path.basename(filename)
    else:
        result_file_name = 'debug_result_' + os.path.basename(filename)
    snapshot.write_videofile(output_path_video + result_file_name, audio=False)
    return

# function which reads in a image file and prepares the frames for the pipeline
def processImage(filename):
    #read in the file and flipping the color from BGR to RGB
    image = cv2.cvtColor(mpimg.imread(filename), cv2.COLOR_BGR2RGB)
    pl = pipeline()
    result = pl(image,1)

    mpimg.imsave( output_path + 'result_' + os.path.basename(filename) , cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    print ('Results saved at ' + output_path + 'result_' + os.path.basename(filename) )

#start-up function - Reads in the program arguments and prepares the respective pipelines
def main():
    # from a conda environment, call 'python main.py <full-path-to-image-file>
    parser = argparse.ArgumentParser(description='Detect lanes and highlight them')
    parser.add_argument('filename')
    parser.add_argument('--debug', dest='debug_mode', action='store_true')
    parser.add_argument('--timeslot', dest='timeslot')
    #read command line agruments
    args = parser.parse_args()
    config.debug_mode = args.debug_mode

    time_count = args.timeslot
    start = 0
    end = 5
    time = []

    if time_count is not None:
        time = [int(i) for i in time_count.split('-') if i.isdigit()]
        if len(time) == 2:
            start = time[0]
            end = time[1]
    else:
        start = -1
        end = -1

    if args.filename[-4:] == '.mp4':
        processVideo(args.filename, start, end)
    else: ##assuming image
        processImage (args.filename)

if __name__ == '__main__':
    main()
