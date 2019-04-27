import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Loading the camera parameters calculated during camera calibration
cam_parameters = np.load('calibration.npz')
# Camera Matrix
k = cam_parameters['mtx']
newK = cam_parameters['newCamMat']
# print('The camera Matrix: \n', k)
# Distortion Coefficients
dist = cam_parameters['dist']

# Loading the stereo pair
left_img = cv.imread('Desk_L.JPG')
right_img = cv.imread('Desk_R.JPG')

# Shape of the left image
R, C, B = left_img.shape

# Converting the images to grayscale
gray_L = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
gray_R = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)

# Using the SIFT feature matching technique from the Assignment 2.
# More explaination of this code part has been given in the report.
# Create object of SIFT
sift = cv.xfeatures2d.SIFT_create()

# Detect the keypoints
kp_L, des_L = sift.detectAndCompute(gray_L, None)
kp_R, des_R = sift.detectAndCompute(gray_R, None)

# FLANN Parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_L, des_R,k=2)

good = []
matched_L = []
matched_R = []

for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        matched_L.append(kp_L[m.queryIdx].pt)
        matched_R.append(kp_R[m.trainIdx].pt)

# Converting the matched keypoints into an array so that it can be passed in a function to draw epilines
# that takes input as a array
matched_L = np.array(matched_L)
matched_R = np.array(matched_R)
print('Shape of matched keypoints: ', matched_L.shape)

#Criteria is defined as follows criteria = (type, number of iterations, accuracy). In this case we are telling the
# algorithm that we care both about number of iterations and accuracy (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER)
# and we selected 30 iterations and an accuracy of 1e-6
# Criteria
crit_cal = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

# Adding New axis to the matched points. This is done so that the points have a dimension of 1 x N x 2.
# This dimension is required because the next opencv function used in line 70 takes these points as the input with this
# particular dimension of 1 x N x 2
matched_L_na = matched_L[:, np.newaxis, :]
matched_R_na = matched_R[:, np.newaxis, :]

# Undistorting the matched key points
# The next step is to multiply the matched keypoints with the inverse camera matrix, undidtort the points and then
# multiply again with the camera matrix.
# In the case of lens distortion, the equations are non-linear and depend on 3 to 8 parameters (k1 to k6, p1 and p2).
# Hence, it would normally require a non-linear solving algorithm (e.g. Newton's method, Levenberg-Marquardt algorithm,
# etc) to inverse such a model and estimate the undistorted coordinates from the distorted ones.
# And this is what is used behind function undistortPoints, with tuned parameters making the optimization fast but a
# little inaccurate.
# The explaination of the function and its input arguments are given in the report
undist_pts_L = cv.undistortPointsIter(matched_L_na, k, dist, R = None, P = newK, criteria=crit_cal)
undist_pts_R = cv.undistortPointsIter(matched_R_na, k, dist, R = None, P = newK, criteria=crit_cal)
print('Shape of the returned Undistorted points: ', undist_pts_L.shape)


# Computing the Fundamental Matrix.

# The fundamental matrix maps points in one stereo image to epipolar lines in the other. It can be computed usingcorresponding
# points in two images recovered in the feature matching stage. In particular, cv2.findFundamentalMat() implements just
# this approach.

# The undistorted keypoints are used for calculating the fundamental matrix. This not only reverses the effect of lens
# distortion, but also transforms the coordinates to normalized image coordinates. Normalized image coordinates
# (not to be confused with the normalization done in the 8-point algorithm) are camera agnostic coordinates that do
# not depend on any of the camera intrinsic parameters. They represent the angle of the bearing vector to the point
# in the real world. For example, a normalized image coordinate of (1, 0) would correspond to a bearing angle of 45
# degrees from the optical axis of the camera in the x direction and 0 degrees in the y direction.

# FINDFUNDAMENTALMAT  Calculates a fundamental matrix from the corresponding points in two images
#
#      F = cv.findFundamentalMat(points1, points2)
#      [F, mask] = cv.findFundamentalMat(...)
#      [...] = cv.findFundamentalMat(..., 'OptionName', optionValue, ...)
#
#  ## Input
#  * __points1__ Cell array of N points from the first image, or numeric array
#    Nx2/Nx1x2/1xNx2. The point coordinates should be floating-point (single or
#    double precision).
#  * __points2__ Cell array or numeric array of the second image points of the
#    same size and format as `points1`.
#
#  ## Output
#  * __F__ Fundamental matrix, 3x3 (or 9x3 in some cases).
#  * __mask__ Optional output mask set by a robust method (RANSAC or LMedS),
#    indicates inliers. Vector of same length as number of points.
#
#  ## Options
#  * __Method__ Method for computing a fundamental matrix. One of:
#    * __7Point__ for a 7-point algorithm. `N = 7`.
#    * __8Point__ for an 8-point algorithm. `N >= 8`.
#    * __Ransac__ for the RANSAC algorithm. `N >= 8`. (default)
#      It needs at least 15 points. 7-point algorithm is used.
#    * __LMedS__ for the LMedS least-median-of-squares algorithm. `N >= 8`.
#      7-point algorithm is used.
#  * __RansacReprojThreshold__ Parameter used only for RANSAC. It is the
#    maximum distance from a point to an epipolar line in pixels, beyond which
#    the point is considered an outlier and is not used for computing the final
#    fundamental matrix. It can be set to something like 1-3, depending on the
#    accuracy of the point localization, image resolution, and the image noise.
#    default 3.0
#  * __Confidence__ Parameter used for the RANSAC and LMedS methods only. It
#    specifies a desirable level of confidence (probability) that the estimated
#    matrix is correct. In the range 0..1 exclusive. default 0.99
#
#  The epipolar geometry is described by the following equation:
#
#      [p2;1]^T * F * [p1;1] = 0
#
#  where `F` is a fundamental matrix, `p1` and `p2` are corresponding points in
#  the first and the second images, respectively.
#
#  The function calculates the fundamental matrix using one of four methods
#  listed above and returns the found fundamental matrix. Normally just one
#  matrix is found. But in case of the 7-point algorithm, the function may
#  return up to 3 solutions (9x3 matrix that stores all 3 matrices
#  sequentially).
#
#  The calculated fundamental matrix may be passed further to
#  cv.computeCorrespondEpilines that finds the epipolar lines corresponding
#  to the specified points. It can also be passed to
#  cv.stereoRectifyUncalibrated to compute the rectification transformation.

F, mask = cv.findFundamentalMat(undist_pts_L, undist_pts_R, cv.FM_RANSAC, ransacReprojThreshold=0.7, confidence=0.99)
print('Fundamental Matrix: \n', F)


# Finding the Camera Projection Matrix:
mtxx= np.array([[3.37289677e+03, 0.00000000e+00, 1.59491773e+03], # this is the camera matrix
 [0.00000000e+00 ,3.37219712e+03, 1.86830009e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

mtxx2= np.array ([[-0.94946855 , 0.3114353,  -0.03895557, 0.98298674], #this is [rotation | translation]
 [-0.29010366, -0.91818082 ,-0.26978482, -0.02762764],
 [-0.11978878 ,-0.24485105 , 0.96213233, 0.18158683]])

P_R= np.matmul( mtxx,mtxx2 )
print('Right Projection matrix=', P_R)


mtxx3= np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,0]])

P_L= np.matmul(mtxx,mtxx3)
print('Left Projection matrix=', P_L)



# Computing Inliner Points. The mask is applied on the matched keypoints that comply with the fundamental matrix.
inLeft = matched_L[mask.ravel() == 1]
inRight = matched_R[mask.ravel() == 1]
print('Shape inRight: ', inRight.shape)
npts= inLeft.shape[0]


# Function to draw the epilines
def draw_epilines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape[:2]
    for r,pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(np.int32(pt1)), 5, color, -1)
        img2 = cv.circle(img2, tuple(np.int32(pt2)), 5, color, -1)
    return img1, img2


# Undistorting the original images
# Basically, the algorithm behind function undistort is the following. For each pixel of the destination lens-corrected
# image do:
#
# 1. Convert the pixel coordinates (u_dst, v_dst) to normalized coordinates (x', y') using the inverse of the calibration matrix K.
# 2. Apply the lens-distortion model, as displayed above, to obtain the distorted normalized coordinates (x'', y'').
# 3. Convert (x'', y'') to distorted pixel coordinates (u_src, v_src) using the calibration matrix K.
# 4. Use the interpolation method of your choice to find the intensity/depth associated with the pixel coordinates
# (u_src, v_src) in the source image, and assign this intensity/depth to the current destination pixel.
left_img_undist  = cv.undistort(left_img, k, dist, None, newK)
right_img_undist = cv.undistort(right_img, k, dist, None, newK)
plt.imsave('Undistorted_Desk_L', left_img_undist)
plt.imsave('Undistorted_Desk_R', right_img_undist)

# Find epilines corresponding to points in right image (second image)
# drawing its lines on the left image
lines_L = cv.computeCorrespondEpilines(inRight.reshape(-1, 1, 2), 2, F)
lines_L = lines_L.reshape(-1, 3)
#left_img_lines, img6 = draw_epilines(gray_L, gray_R, lines_L, inLeft, inRight)
left_img_lines, img6 = draw_epilines(left_img_undist.copy(), right_img_undist.copy(), lines_L, inLeft, inRight)

# Finding the epilines corresponding to points in left image (First Image) and
# drawing its lines on the right image
# COMPUTECORRESPONDEPILINES  For points in an image of a stereo pair, computes the corresponding epilines in the other image
#
#      lines = cv.computeCorrespondEpilines(points, F)
#      [...] = cv.computeCorrespondEpilines(..., 'OptionName', optionValue, ...)
#
#  ## Input
#  * __points__ Input points. Nx2/Nx1x2/1xNx2 floating-point array, or cell
#    array of length N of 2-element vectors `{[x,y], ...}`.
# % * __F__ 3x3 Fundamental matrix that can be estimated using
#    cv.findFundamentalMat or cv.stereoRectify.
#
#  ## Output
#  * __lines__ Output vector of the epipolar lines corresponding to the points
#    in the other image. Each line `ax + by + c = 0` is encoded by 3 numbers
#    `(a,b,c)`. Nx3/Nx1x3 numeric matrix or a cell-array of 3-element vectors
#    `{[a,b,c], ...}` depending on `points` format.
#
#  ## Options
#  * __WhichImage__ Index of the image (1 or 2) that contains the points.
#   default 1.
#
#  For every point in one of the two images of a stereo pair, the function
#  finds the equation of the corresponding epipolar line in the other image.
#
#  From the fundamental matrix definition , line
#  `lines2{i}` in the second image for the point `points1{i}` in the first image
#  (when `WhichImage=1`) is computed as:
#
#      lines2{i} = F * points1{i}
#
#  And vice versa, when `WhichImage=2`, `lines1{i}` is computed from
#  `points2{i}` as:
#
#      lines1{i} = F^T * points2{i}
#
#  Line coefficients are defined up to a scale. They are normalized so that
#  `a_i^2 + b_i^2 = 1`.

lines_R = cv.computeCorrespondEpilines(inLeft.reshape(-1, 1, 2), 1, F)
lines_R = lines_R.reshape(-1, 3)
#right_img_lines, img4 = draw_epilines(gray_R, gray_L, lines_R, inRight, inLeft)
right_img_lines, img4 = draw_epilines(right_img_undist.copy(), left_img_undist.copy(), lines_R, inRight, inLeft)

cv.imwrite('Epilines_Desk_L.bmp', left_img_lines)
cv.imwrite('Epilines_Desk_R.bmp', right_img_lines)

new_right = inRight.reshape(-1, 1, 2)
new_left = inLeft.reshape(-1, 1, 2)
#npts= new_left.shape[0]

# Computing the homography matrices

# STEREORECTIFYUNCALIBRATED  Computes a rectification transform for an uncalibrated stereo camera
#
#      [H1,H2,success] = cv.stereoRectifyUncalibrated(points1, points2, F, imageSize)
#      [...] = cv.stereoRectifyUncalibrated(..., 'OptionName', optionValue, ...)
#
#  ## Input
#  * __points1__ Array of feature points in the first image as a cell array of
#    2-element vectors: `{[x1, y1], [x2, y2], ...}` or an Nx2/Nx1x2/1xNx2
#    numeric array. The same formats as in cv.findFundamentalMat are supported.
#  * __points2__ The corresponding points in the second image, same size and
#    type as `points1`.
#  * __F__ Input 3x3 fundamental matrix. It can be computed from the same set
#    of point pairs using cv.findFundamentalMat.
#  * __imageSize__ Size of the image `[w,h]`.
#
#  ## Output
#  * __H1__ 3x3 rectification homography matrix for the first image.
#  * __H2__ 3x3 rectification homography matrix for the second image.
#  * __success__ success flag. Returns true if successfull, false otherwise.
#
#  ## Options
#  * __Threshold__ Optional threshold used to filter out the outliers. If the
#    parameter is greater than zero, all the point pairs that do not comply
#    with the epipolar geometry (that is, the points for which
#    `|points2{i}' * F * points1{i}| > Threshold`) are rejected prior to
#    computing the homographies. Otherwise, all the points are considered
#    inliers. default 5
#
#  The function computes the rectification transformations without knowing
#  intrinsic parameters of the cameras and their relative position in the
#  space, which explains the suffix "uncalibrated". Another related difference
#  from cv.stereoRectify is that the function outputs not the rectification
#  transformations in the object (3D) space, but the planar perspective
#  transformations encoded by the homography matrices `H1` and `H2`. The
#  function implements the algorithm [Hartley99].
#
#  ### Note
#  While the algorithm does not need to know the intrinsic parameters of the
#  cameras, it heavily depends on the epipolar geometry. Therefore, if the
#  camera lenses have a significant distortion, it would be better to correct
#  it before computing the fundamental matrix and calling this function. For
#  example, distortion coefficients can be estimated for each head of stereo
#  camera separately by using cv.calibrateCamera. Then, the images can be
#  corrected using cv.undistort, or just the point coordinates can be corrected
#  with cv.undistortPoints.
ret, Hl, Hr = cv.stereoRectifyUncalibrated(new_left[:, :, 0:2], new_right[:, :, 0:2], F, (C, R))

Hl/=Hl[2,2]
Hr/=Hr[2,2]
Hl[0,2]+=250

# Printing the Homographies
print('Left Homography: \n', Hl)
print('The Right Homography: \n',Hr)

# Rectifying the Images:
# WARPPERSPECTIVE  Applies a perspective transformation to an image
#
#      dst = cv.warpPerspective(src, M)
#      dst = cv.warpPerspective(src, M, 'OptionName',optionValue, ...)
#
#  ## Input
#  * __src__ Input image.
#  * __M__ 3x3 transformation matrix, floating-point.
#
#  ## Output
#  * __dst__ Output image that has the size `DSize` (with
#    `size(dst,3) == size(src,3)`) and the same type as `src`.
#
#  ## Options
#  * __DSize__ Size of the output image `[w,h]`. Default `[0,0]` means using
#    the same size as the input `[size(src,2) size(src,1)]`.
# * __Interpolation__ interpolation method, default 'Linear'. One of:
#   * __Nearest__ nearest neighbor interpolation
#    * __Linear__ bilinear interpolation
#    * __Cubic__ bicubic interpolation
#    * __Lanczos4__ Lanczos interpolation over 8x8 neighborhood
#  * __WarpInverse__ Logical flag to apply inverse perspective transform,
#    meaning that `M` is the inverse transformation (`dst -> src`).
#    default false
#  * __BorderType__ Pixel extrapolation method. When 'Transparent', it means
#    that the pixels in the destination image corresponding to the "outliers"
#    in the source image are not modified by the function. default 'Constant'
#    * __Constant__ `iiiiii|abcdefgh|iiiiiii` with some specified `i`
#    * __Replicate__ `aaaaaa|abcdefgh|hhhhhhh`
#    * __Reflect__ `fedcba|abcdefgh|hgfedcb`
#    * __Reflect101__ `gfedcb|abcdefgh|gfedcba`
#    * __Wrap__ `cdefgh|abcdefgh|abcdefg`
#   * __Transparent__ `uvwxyz|abcdefgh|ijklmno`
#    * __Default__ same as 'Reflect101'
#  * __BorderValue__ Value used in case of a constant border. default 0
#  * __Dst__ Optional initial image for the output. If not set, it is
#    automatically created by the function. Note that it must match the
#    expected size `DSize` and the type of `src`, otherwise it is ignored and
#    recreated by the function. This option is only useful when
#    `BorderType=Transparent`, in which case the transformed image is drawn
#    onto the existing `Dst` without extrapolating pixels. Not set by default.
#
#  The function cv.warpPerspective transforms the source image using the
#  specified matrix:
#
#      dst(x,y) = src((M_11*x + M_12*y + M_13) / (M_31*x + M_32*y + M_33),
#                     (M_21*x + M_22*y + M_23) / (M_31*x + M_32*y + M_33))
#
#  when the `WarpInverse` option is true. Otherwise, the transformation is first
#  inverted with cv.invert and then put in the formula above instead of `M`.
rect_left = cv. warpPerspective(left_img_undist, Hl, (C, R))
rect_right = cv.warpPerspective(right_img_undist, Hr, (C,R))

cv.imwrite('Rect_Left_desk.bmp', rect_left)
cv.imwrite('Rect_right_desk.bmp', rect_right)

# Finding the Essential Matrix
# The Detailed method of finding the Essential matrix and the idea behind this algorithm is explained in the report.
E= np.matmul(np.transpose(newK), np.matmul(F,newK))
# Computing Singular Value Decomposition:
U,S,V = np.linalg.svd(E)
s= (S[0] + S[1])/ 2
S[0] = S[1] = s
S[2]= 0
S=np.diag(S)
E= np.matmul(U, np.matmul(S, np.transpose(V)))
print('Essential Matrix=', E)
np.savez('Essentialmatrix.npz', E)


# Finding the rotation matrix and translation vector from the Essential Matrix

# RECOVERPOSE  Recover relative camera rotation and translation from an estimated essential matrix and the corresponding points in two images, using cheirality check
#
#      [R, t, good] = cv.recoverPose(E, points1, points2)
#      [R, t, good, mask, triangulatedPoints] = cv.recoverPose(...)
#      [...] = cv.recoverPose(..., 'OptionName', optionValue, ...)
#
#  ## Input
#  * __E__ The input essential matrix, 3x3.
#  * __points1__ Cell array of N 2D points from the first image, or numeric
#    array Nx2/Nx1x2/1xNx2. The point coordinates should be floating-point
#    (single or double precision).
#  * __points2__ Cell array or numeric array of the second image points of the
#    same size and format as `points1`.
#
#  ## Output
#  * __R__ Recovered relative rotation, 3x3 matrix.
#  * __t__ Recovered relative translation, 3x1 vector.
#  * __good__ the number of inliers which pass the cheirality check.
#  * __mask__ Output mask for inliers in `points1` and `points2`. In the output
#    mask only inliers which pass the cheirality check. Vector of length N, see
#    the `Mask` input option.
#  * __triangulatedPoints__ 3D points which were reconstructed by triangulation
#
#  ## Options
# * __CameraMatrix__ Camera matrix `K = [fx 0 cx; 0 fy cy; 0 0 1]`. Note that
#   this function assumes that `points1` and `points2` are feature points from
#    cameras with the same camera matrix. default `eye(3)`.
#  * __DistanceThreshold__ threshold distance which is used to filter out far
#   away points (i.e. infinite points). default 50.0
#  * __Mask__ Input mask of length N for inliers in `points1` and `points2`
#    (0 for outliers and to 1 for the other points (inliers). If it is not
#    empty, then it marks inliers in `points1` and `points2` for then given
#   essential matrix `E`. Only these inliers will be used to recover pose.
#    Not set by default.
#
#  This function decomposes an essential matrix using cv.decomposeEssentialMat
#  and then verifies possible pose hypotheses by doing cheirality check. The
#  cheirality check basically means that the triangulated 3D points should have
#  positive depth.
#
#  This function can be used to process output `E` and `mask` from
#  cv.findEssentialMat. In this scenario, `points1` and `points2` are the same
#  input for cv.findEssentialMat.
ret, R, t, mask, TriangulatedPoints = cv.recoverPose(E,new_left,new_right,newK, distanceThresh=1000)
print ('rot=', R)
print('trans=', t)
print('points=', TriangulatedPoints)
points = cv.convertPointsFromHomogeneous(np.transpose(TriangulatedPoints))
# CONVERTPOINTSFROMHOMOGENEOUS  Converts points from homogeneous to Euclidean space
#
#      dst = cv.convertPointsFromHomogeneous(src)
#
#  ## Input
#  * __src__ Input vector of N-dimensional points (3D/4D points).
#    Mx3/Mx1x3/1xMx3 or Mx4/Mx1x4/1xMx4 numeric array, or cell-array of
#    3/4-element vectors in the form: `{[x,y,z], [x,y,z], ...}` or
#    `{[x,y,z,w], [x,y,z,w], ...}`. Supports floating-point types.
#
#  ## Output
#  * __dst__ Output vector of (N-1)-dimensional points (2D/3D points).
#    Mx2/Mx1x2 or Mx3/Mx1x3 numeric array, or cell-array of 2/3-elements
#    vectors, respectively matching the input shape.
#
#  The function converts points homogeneous to Euclidean space using
#  perspective projection. That is, each point `(x1, x2, ..., x(n-1), xn)` is
#  converted to `(x1/xn, x2/xn, ..., x(n-1)/xn)`. When `xn=0`, the output point
#  coordinates will be `(0,0,0,...)`.
np.savez('Pose.npz', rot=R, trans=t, point=points)

# Compute Projection Error:

# RODRIGUES  Converts a rotation matrix to a rotation vector or vice versa
#
#      dst = cv.Rodrigues(src)
#      [dst,jacobian] = cv.Rodrigues(src)
#
#  ## Input
#  * __src__ Input rotation vector (3x1 or 1x3) or rotation matrix (3x3). Both
#    single and double-precision floating-point types are supported.
#
#  ## Output
#  * __dst__ Output rotation matrix (3x3) or rotation vector (3x1 or 1x3),
#    respectively. Same data type as `src`.
#  * __jacobian__ Optional output Jacobian matrix, 3x9 or 9x3, which is a
#    matrix of partial derivatives of the output array components with respect
#    to the input array components. Same data type as `src`.
#
#  The function transforms a rotation matrix in the following way:
#
#      theta <- norm(r)
#     r <- r/theta
#      R = cos(theta) * I + (1 - cos(theta)) * r * r^T + sin(theta) * A
#      A = [0, -rz, ry; rz, 0, -rx; -ry, rx, 0]
#
#  Inverse transformation can be also done easily, since
#
#      sin(theta) * A = (R - R^T) / 2
#
#  A rotation vector is a convenient and most compact representation of a
#  rotation matrix (since any rotation matrix has just 3 degrees of
#  freedom). The representation is used in the global 3D geometry
#  optimization procedures like cv.calibrateCamera, cv.stereoCalibrate, or
#  cv.solvePnP.
rotVec= cv.Rodrigues(R)
print('rotVec = ', rotVec[0])
repPts_left= cv.projectPoints(points,rotVec[0], t, newK, dist)
repPts_right = cv.projectPoints( points, rotVec[0], t, newK, dist)

# PROJECTPOINTS  Projects 3D points to an image plane
#
#      imagePoints = cv.projectPoints(objectPoints, rvec, tvec, cameraMatrix)
#      [imagePoints, jacobian] = cv.projectPoints(...)
#      [...] = cv.projectPoints(..., 'OptionName', optionValue, ...)
#
#  ## Input
#  * __objectPoints__ Array of object points, Nx3/Nx1x3/1xNx3 array or cell
#    array of 3-element vectors `{[x,y,z],...}`, where `N` is the number of
#    points in the view.
#  * __rvec__ Rotation vector or matrix (3x1/1x3 or 3x3). See cv.Rodrigues for
#    details below.
#  * __tvec__ Translation vector (3x1/1x3).
#  * __cameraMatrix__ Camera matrix 3x3, `A = [fx 0 cx; 0 fy cy; 0 0 1]`.
#
#  ## Output
#  * __imagePoints__ Output array of image points, Nx2/Nx1x2/1xNx2 array or
#    cell array of 2-element vectors `{[x,y], ...}`.
#  * __jacobian__ Optional output `(2N)x(3+3+2+2+numel(DistCoeffs))` jacobian
#    matrix of derivatives of image points with respect to components of the
#    rotation vector (3), translation vector (3), focal lengths (2),
#    coordinates of the principal point (2), and the distortion coefficients
#    (`numel(DistCoeffs)`).
#
#  ## Options
#  * __DistCoeffs__ Input vector of distortion coefficients
#    `[k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy]` of 4, 5, 8, 12 or 14
#    elements. If the vector is empty, the zero distortion coefficients are
#    assumed. default empty
#  * __AspectRatio__ Optional "fixed aspect ratio" parameter. If the parameter
#    is not 0, the function assumes that the aspect ratio (`fx/fy`) is fixed
#    and correspondingly adjusts the jacobian matrix. default 0.
#
#  The function computes projections of 3D points to the image plane given
#  intrinsic and extrinsic camera parameters. Optionally, the function
#  computes Jacobians - matrices of partial derivatives of image points
#  coordinates (as functions of all the input parameters) with respect to
#  the particular parameters, intrinsic and/or extrinsic. The Jacobians are
#  used during the global optimization in cv.calibrateCamera, cv.solvePnP,
#  and cv.stereoCalibrate. The function itself can also be used to compute a
#  re-projection error given the current intrinsic and extrinsic parameters.
#
#  ### Note
#  By setting `rvec=tvec=[0,0,0]` or by setting `cameraMatrix` to a 3x3
#  identity matrix, or by passing zero distortion coefficients, you can get
#  various useful partial cases of the function. This means that you can
#  compute the distorted coordinates for a sparse set of points or apply a
#  perspective transformation (and also compute the derivatives) in the
#  ideal zero-distortion setup.


# # Compare the reprojected points to the undistorted original points by computing the average euclidean norm
# # of the differences
dlx= np.squeeze(repPts_left[0][:,0,0] - new_left[:,0,0])
drx= np.squeeze(repPts_right[0][:,0,0] - new_right[:,0,0])
dly= np.squeeze(repPts_left[0][:,0,1] - new_left[:,0,1])
dry= np.squeeze(repPts_right[0][:,0,1] - new_right[:,0,1])

error_L_x= np.sum(np.abs(dlx))/ npts
error_R_x= np.sum(np.abs(drx))/ npts
error_L_y= np.sum(np.abs(dly))/ npts
error_R_y= np.sum(np.abs(dry))/ npts

print('error_Left_x=', error_L_x)
print('error_Right_x=', error_R_x)
print('error_Left_y=', error_L_y)
print('error_Right_y=', error_R_y)

# Computing the disparity map of the images
# Setting the Disparity parameters

# STEREOSGBM  Class for computing stereo correspondence using the semi-global block matching algorithm
#      * By default, the algorithm is single-pass, which means that you
#        consider only 5 directions instead of 8. Set `Mode='HH'` in the
#        contructor to run the full variant of the algorithm but beware that it
#        may consume a lot of memory.
#
#      * The algorithm matches blocks, not individual pixels. Though, setting
#        `BlockSize=1` reduces the blocks to single pixels.
#
#      * Mutual information cost function is not implemented. Instead, a
#        simpler Birchfield-Tomasi sub-pixel metric from [BT96] is used.
#        Though, the color images are supported as well.
#
#      * Some pre- and post- processing steps from K. Konolige algorithm
#        cv.StereoBM are included, for example: pre-filtering (`XSobel` type)
#        and post-filtering (uniqueness check, quadratic interpolation and
#        speckle filtering).

win_size = 5
min_disp = 0
max_disp = 64
num_disp = max_disp - min_disp

# Creating the Block Matching Object
stereo = cv.StereoSGBM_create(minDisparity=min_disp, blockSize=2,
                              uniquenessRatio=5, speckleWindowSize=5, speckleRange=5,
                              disp12MaxDiff=1, P1 =8*3*win_size**2,
                              P2=32*3*win_size**2)
right_matcher = cv.ximgproc.createRightMatcher(stereo)

wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(80000)
wls_filter.setSigmaColor(1.2)

disparity_left  = stereo.compute(rect_right, rect_left)
disparity_right = right_matcher.compute(rect_left, rect_right)
disparity_left  = np.int16(disparity_left)
disparity_right = np.int16(disparity_right)
filteredImg = wls_filter.filter(disparity_left, rect_left, None, disparity_right)

depth_map = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
depth_map = np.uint8(depth_map)
#depth_map = cv.bitwise_not(depth_map) # Invert image. Optional depending on stereo pair
imcolor= cv.applyColorMap(depth_map, cv.COLORMAP_JET)
plt.imsave("depth.png", imcolor)


