# Epipolar-Stereopsis

The step wise process with detailed description of each step of this project is explained below:

1. Take a stereo pair (like Desk_L.JPG and Desl_R.JPG) and generate a set of matched keypoints. To do so use the python file ORB.py. ORB is a feature matching technique in computer vision.
   The generated keypoints are displayed in the image Matches_Stereo.png.
   
2. (See the Camera-Calibration repository before moving forward). By using the camera matrix and distortion coefficients obtained via camera calibration.
    we undistort the keypoints and stereo images. The undistorted images are named as Undistorted_Desk_L.png and Undistorted_Desk_R.png
    
3. Fundamental matrix is generated using undistorted keypoints. The epipolar lines are then drawn on the stereo image pair. The images are Epilines_L.PNG and Epilines_R.PNG

4. Camera projection matrices are generated.

5. Triangulation is used to estimate the 3D positions of the keypoints in the scene. 

6. Left and right homography matrics are generated. 

7. Using the homography matrices, both the images were rectified. The images are named as Rect_Left.bmp and Rect_Right.bmp.

8. To check the correct rectification, ORB feature matching is used to ganerate parallel matches in the images. The image is saved as Matches_Rectified.png.

9. The rectified images are used to generate a disparity map.

The code for all above steps can be found in Epilolar Stereopsis.py

If you find and problems, please feel free to contact me on vedansh.thakkar@vanderbilt.edu
