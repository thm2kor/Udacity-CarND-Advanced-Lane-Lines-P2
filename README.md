# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![Final Result](./output_videos/result_project_video.gif)
---
## Objective
This project aims to identify and track lanes in more challenging highway scenarios (with lighting changes, color changes, shadows, underpasses) using a single camera mounted on the center of a vehicle.

[//]: # (Image References)

[image1]: ./output_images/calibration2_corners.jpg "identified corners"
[image2]: ./output_images/calibration2_undistored.jpg "Undistored chessboard image"
[image3]: ./output_images/test6_undistored.jpg "Undistored pipeline image"
[image4]: ./output_images/straight_lines1_warped_cmp.jpg "Warped image - Straight road"
[image5]: ./output_images/test6_warped_cmp.jpg "Warped image - Curved road"
[image6]: ./output_images/results_gradient_thresholds.jpg "Gradient+S binary Performance"
[image7]: ./output_images/results_color_thresholds.jpg "Color binary Performance"
[video1]: ./output_videos/debug_result_project_video.gif "Debug mode - Project video"
[video2]: ./output_videos/result_project_video.mp4 "Final - Project video"

---

## Overview of Files
| File| Description | Supported flags |
| ------ | ------ | ----- |
| config.py | global parameters for the project | No flags |
| calibrations.py | code for calibrating and undistorting images | -- corners, --undistort, --filename=RELATIVE_PATH_TO_IMAGE |
| transforms.py | code for perspective transformation| --warp=RELATIVE_PATH_TO_IMAGE, --unwarp=RELATIVE_PATH_TO_IMAGE, --comp |
| thresholds.py | code for binary thresholding of images| No flags |
| lanes.py | code for lane detection and lane line state management | No flags |
| main.py | code for pipeline for images and video | RELATIVE_PATH_TO_IMAGE --debug --timeslot=1-10 |


## Camera Calibration
The first step is the correction for the effect of image distortion. These distortions are caused by the angle of light and the position of the lens while capturing the image. The distortions changes the size and shapes of objects in an image. Calibration is a process which measures and corrects the distortion errors based on measurements from standard shapes. In this project, we use a chessboard to calibrate the camera. The regular high contrast pattern makes it an ideal candidate for calibrating a camera.

### Compute camera calibration matrix and distortion coefficients
The project repository provided by Udacity contained a set of [calibration images](./camera_cal/) which were taken at different angles and distances. For each calibration image, the following steps were performed

1. Identify the 3D real world points for each of the identified corners using the OpenCV function `cv2.findChessboardCorners( ... )`. These points are called **object points**
2. The corresponding 2D coordinates of these points in the image are the locations where two black squares touch each other in the respective chess boards. These points are called **image points**.
3. With the identified object points and image points, calibrate the camera using the function `cv2.calibrateCamera( ... )` which returns the **camera matrix and distortion coefficients**
4. The object points, image points, camera matrix and distortion coefficients for each image are converted to a byte stream and saved to a **[pickle file](./camera_cal/camera_distortion_pickle.p)**

The above steps are implemented in the [calibrations.py](./calibrations.py) file. The results could be verified by running the following command from a conda environment:

```sh
$ python calibrations.py --corners
```

For each calibration*.jpg file in the [calibration folder](./camera_cal/), a corresponding result file with the postfix *_corners.jpg* is created in the [output_images folder](./output_images). The result files shows a side-by-side view of the original chessboard image with distortion and the resulting image with the identified corners. A sample output for one of the calibration image is shown below:
![alt text][image1]

As a side note, some of the chessboard images could not be calibrated because `cv2.findChessboardCorners` was unable to detect the desired number of internal corners.

### Apply distortion correction to raw images
This step unpickles the camera matrix and distortion co-efficients from the previously cached **[pickle file](./camera_cal/camera_distortion_pickle.p)** . The raw images are calibrated using the openCV `cv2.undistort(image, mtx, dist, None, None)` function. The output could be verified by running the following command from a conda environment:

```sh
$ python calibrations.py --undistort
```

For each calibration*.jpg file in the [calibration folder](./camera_cal), a corresponding result file with the postfix *_undistorted.jpg* is created in the [output_images folder](./output_images). The result files shows a side-by-side view of the original chessboard image with distortion and the resulting undistorted image. A sample output for one of the calibration image is shown below:
![alt text][image2]

---

## Pipeline (Image file)
The lane detection pipeline consists of the following stages:
1. Distortion correction
2. Perspective transformation  
3. Binary thresholding
4. Lane line detecting
5. Calculating radius of curvature
6. Calculating vehicle position


### Distortion correction
The calibration routines discussed in the previous chapter are encapsulated in the [`class cameraCalibration`](./calibrations.py). A given image is undistorted by simply initializing a cameraCalibration object and a subsequent call to `cameraCalibration::undistort(filename)` function.
The distortion correction on a given image can be executed by the following command:

```sh
$ python calibrations.py --filename=<RELATIVE_PATH_TO_IMAGE>
```
An example calibration by `python calibrations.py --filename=test_images\test6.jpg` gives the following result:
![alt text][image3]

### Perspective transformation
The undistorted image seen above exhibits the so-called *perspective phenomenon* where objects farther on the road appears smaller and parallel lines seem to converge to a point. Perspective transform is the step which warps an image by transforming the apparent z-coordinates of the object points. This effectively adapts the objects 2D representation.
Perspective transformation involves the following steps:
1. Define the source points (`src`) that define a  rectangle in a given image
2. Define the destination image points (`dst`) on the transformed/warped image
3. Map the `src` and `dst` points by the function `cv2.getPerspectiveTransform( ... )` to derive the **mapping perspective matrix - M**
4. Apply the tranformation matrix *M* on the original undistorted image to derive the warped image by calling the `cv2.warpPerspective ( ...)` function.
5. To reverse the perspective transform, the `src` and `dst` points needs to be swapped to derive the the **MInv matrix**  using the `cv2.getPerspectiveTransform( ... )` function.
6. A call to `cv2.warpPerspective ( ...)` with the `MInv` matrix unwarps the image

The above steps are encapsulated in the [`class perspectiveTranform`](./transforms.py). The perspective transformation on a given image can be performed by executing the following command:

```sh
$ python transforms.py --warp=<RELATIVE_PATH_TO_IMAGE> --comp
$ python transforms.py --unwarp=<RELATIVE_PATH_TO_IMAGE> --comp

```
A sample transformation on a straight and curved road are shown below:
![alt text][image4]
![alt text][image5]

### Binary thresholding - Edge detection
To efficiently extract the lane lines, several thresholding techniques were discussed in the Lesson 7: Gradient and color spaces.
An explorative study on several combination of sobel gradient thresholds, magnitude thresholds, color thresholds in different color space like RGB, HSV, Luv and Lab were carried out.
The standard combination of Sobel gradients and S-binary thresholds gave reasonably good results for detecting yellow and white lines. But they failed to detect the lines on roads with less contrast and scenarios with shadows.
The performance of the Sobel gradient and S-binary thresholds could be seen in the below picture.
![alt text][image6]

For a robust performance under shadows and different lighting scenarios, I explored color spaces other than the HSV and HSL color space. I took help from the Udacity Mentor network. I got a [hint](https://knowledge.udacity.com/questions/32588) that the b channel from Lab color space and l channel from Luv color space with a specific range of thresholds gave good results for yellow and white lines under normal lighting as well as under shadows and low contrast surfaces.
So i decided to do the binary thresholding only on Lab and Luv color spaces. The performance of the color thresholds showed good results:
![alt text][image7]

The code for the binary thresholds are available in the [`class thresholdedImage`](./thresholds.py) . The functions `luv_l_thresh` and `lab_b_thresh` extracts the l and b channels respectively, applies the given thresholds and returns the thresholded image. The function `applyThresholds` OR's the output from the two functions and returns the final binary thresholded image.
An extract of the code is shown below:
```python
# Returns the binary image combining (OR) the binary thresholded l channel
# from the Luv color space and b channel from Lab color space

def applyThresholds(self):

    l_binary_output = self.luv_l_thresh(self.image)
    b_binary_output = self.lab_b_thresh(self.image)

    # Combine Luv and Lab B color space channel thresholds
    combined = np.zeros_like(l_binary_output)
    combined[(l_binary_output == 1) | (b_binary_output == 1)] = 1

    return combined
```

### Lane line detecting
After identifying the edges on the warped images, the next step is to identify the potential lane lines from the image by plotting a histogram of the binary pixels on the warped, binary-thresholded image. This serves as a starting position for the lanes. I extensively re-used the code from the Lesson 8(Advanced Computer Vision). As suggested in the lesson, i defined a [`class line`](./lanes.py) which represents the internal state of a lane line. In addition, I defined a [`class drivingLane`](./lanes.py) which encapsulates the detection of the left and right line, validation of the lines and filling of the lanes lines with a defined color.

1. The `track` (instance of `class drivingLane`) object reads in an image and instantiates one line class per detected line.
2. From the first image frame, the `track` objects uses the sliding windows method as discussed in Lesson 8 (Advanced Computer Vision :Finding the Lines: Sliding Window) to detect a set of points (X and Y) which could be a potential lane.
3. The detected points are fit to a second degree polynomial using the function `cv2.polyfit (x, y, 2)` and forwarded to the respective `line` object.
4. The `line` object internally validates the recent fit and adds it to an array of line fit.
5. The `line` object calculates a best-fit based on the weighted average of the line fit array (10 elements). The weights are determined by the count of the pixels which were used for the `cv2.polyfit`.

A sample of lane line detection is shown in the below short video frame. The debug video could be prepared by running the following command:

```sh
$ python main.py test_images\project_video.mp4 --debug --timeslot=1-3
```
![alt text][video1]

In the output frame, the lane pixels for the left line and right line could be identified by the blue and red colors. The fitted line is drawn with a green color.

After the line is detected and validated, it is unwarped and overlayed back onto the original image using the function `cv2.warpPerspective`.
```python
cv2.warpPerspective(image, self.Minv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
```
The code for overlay is available in the module [lanes.py](./lanes.py). The function `overlay_lanes` prepares a the set of x and y in stacked array and uses the function cv2.fillpoly to draw a closed region enclosing the x and y coordinates. In addition, the polynomial fitted lane lines are also drawn with a pre-defined thickness.

### Calculating radius of curvature
Since the radius of curvature is measured in world-space coordinates(metres), I had to normalize the lane dimensions to meters. For this, i took [one](./test_images/straight_lines1_warped.jpg) of the warped straight line image as a reference and manually calculated the horizontal distance between 2 lanes lines and the vertical distance of a lane segment. This pixel sizes corresponded to the real-world values of 3.7m and 3 meters respectively. The `track` object initialize two variable in real-world space as below:

```python
self.ym_per_pix = 3/110 # 110 is the number of pixels for 1 lane segement in straight_line1_warped.jpg
self.xm_per_pix = 3.7/380	# 380 is number of pixels for 1 lane width in straight_line1_warped.jpg
```
With the above normalization values, each lane calculates the radius of curvature by the following steps
1. Normalize the y and x points by multiplying the values by ym_per_pix and xm_per_pix respectively
2. Fit a second-degree polynomial with the real-world values of y and x
3. With the returned coefficients , the radius of coefficient can be calculate with the formula:
```
self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5)/abs(2*fit_cr[0])
```
4. Since the measurement of radius of curvature is done closest to your vehicle, the `y_eval` value corresponding to the bottom of the image is used.

The following code explains the radius of curvature calculations:
```python
    def calc_curavture(self, ym_per_pix, xm_per_pix ):
        ploty = np.linspace(0, (config.IMAGE_HEIGHT)-1, config.IMAGE_HEIGHT)
        #fit a second degree polynomial, which fits the current x and y points
        fit_cr = np.polyfit((self.ally * ym_per_pix), (self.allx * xm_per_pix), 2)
        #fit a second degree polynomial, which fits the current x and y points
        y_eval = np.max(ploty)*ym_per_pix

        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5)/abs(2*fit_cr[0])
        return self.radius_of_curvature
```
### Calculating vehicle position
I took support from the Udacity knowledge portal to calculate the vehicle position. The vehicle position is calculated with the following steps:

1. For calculating the vehicle position, it can be assumed that the camera is mounted at the centre of the vehicle such that the lane center is the midpoint at the bottom of the image.
```python
vehicle_position = image_shape[1]/2
```
2. The center of the lane is calculated as the average of the left line and right line x-intercepts:
```python
leftline_intercept = leftline.best_fit[0]*height**2 + leftline.best_fit[1]*height + leftline.best_fit[2]
rightline_intercept = rightline.best_fit[0]*height**2 + rightline.best_fit[1]*height + rightline.best_fit[2]
lane_center = (leftline_intercept + rightline_intercept) /2

```
3. The vehicle's position from the center is calculated by taking the absolute value of the vehicle position minus the halfway point along the horizontal axis
```python
self.vehicle_pos = (vehicle_position - lane_center) * self.xm_per_pix
```
4. Since the image positions and lane center are in pixel space, it is multiplied with the `xm_per_pix` scaling to convert them to real-world coordinates.

---
### Pipeline video
The link to the output videos could be found [here](./output_videos). The pipeline worked reasonably well for [project_video.mp4](./output_videos/result_project_video.mp4) and [project_video.mp4](./output_videos/result_challenge_video.mp4). Whenever the vehicle comes out of a low contrast road, it wobbles, but recovers after few frames. This shows that my smoothing works well, though i could think of some better techniques for the next projects.

---
