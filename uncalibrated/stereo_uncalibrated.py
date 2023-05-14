"""
Reference: https://stackoverflow.com/questions/36172913/opencv-depth-map-from-uncalibrated-stereo-system

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

imgL = cv2.imread('uncalibrated\imgL\imgL0.png', cv2.IMREAD_GRAYSCALE)  # left image
imgR = cv2.imread('uncalibrated\imgR\imgR0.png', cv2.IMREAD_GRAYSCALE)  # right image
imgL = cv2.GaussianBlur(imgL, (5,5), 0)
imgR = cv2.GaussianBlur(imgR, (5,5), 0)

def get_keypoints_and_descriptors(imgL, imgR):
    """Use ORB detector and FLANN matcher to get keypoints, descritpors,
    and corresponding matches that will be good for computing
    homography.
    """
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(imgL,None)
    kp2, des2 = sift.detectAndCompute(imgR,None)
    des1=np.float32(des1)
    des2=np.float32(des2)
    ############## Using FLANN matcher ##############
    # Each keypoint of the first image is matched with a number of
    # keypoints from the second image. k=2 means keep the 2 best matches
    # for each keypoint (best matches = the ones with the smallest
    # distance measurement).
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann_match_pairs = flann.knnMatch(des1,des2,k=2)
    return kp1, des1, kp2, des2, flann_match_pairs


def lowes_ratio_test(matches, ratio_threshold=0.6):
    """Filter matches using the Lowe's ratio test.

    The ratio test checks if matches are ambiguous and should be
    removed by checking that the two distances are sufficiently
    different. If they are not, then the match at that keypoint is
    ignored.

    https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
    """
    filtered_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            filtered_matches.append(m)
    return filtered_matches


def draw_matches(imgL, imgR, kp1, des1, kp2, des2, flann_match_pairs):
    """Draw the first 8 mathces between the left and right images."""
    # https://docs.opencv.org/4.2.0/d4/d5d/group__features2d__draw.html
    # https://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html
    img = cv2.drawMatches(
        imgL,
        kp1,
        imgR,
        kp2,
        flann_match_pairs[:8],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imshow("Matches", img)
    cv2.imwrite("ORB_FLANN_Matches.png", img)
    cv2.waitKey(0)


def compute_fundamental_matrix(matches, kp1, kp2, method=cv2.FM_RANSAC):
    """Use the set of good mathces to estimate the Fundamental Matrix.

    See  https://en.wikipedia.org/wiki/Eight-point_algorithm#The_normalized_eight-point_algorithm
    for more info.
    """
    pts1, pts2 = [], []
    fundamental_matrix, inliers = None, None
    for m in matches[:8]:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
    if pts1 and pts2:
        # You can play with the Threshold and confidence values here
        # until you get something that gives you reasonable results. I
        # used the defaults
        fundamental_matrix, inliers = cv2.findFundamentalMat(
            np.float32(pts1),
            np.float32(pts2),
            method=method,
            ransacReprojThreshold=3,
            confidence=0.99,
        )
    return fundamental_matrix, inliers, pts1, pts2


############## Find good keypoints to use ##############
kp1, des1, kp2, des2, flann_match_pairs = get_keypoints_and_descriptors(imgL, imgR)
good_matches = lowes_ratio_test(flann_match_pairs, 0.197)
draw_matches(imgL, imgR, kp1, des1, kp2, des2, good_matches)


############## Compute Fundamental Matrix ##############
F, I, points1, points2 = compute_fundamental_matrix(good_matches, kp1, kp2)


############## Stereo rectify uncalibrated ##############
h1, w1 = imgL.shape
h2, w2 = imgR.shape
thresh = 0
_, H1, H2 = cv2.stereoRectifyUncalibrated(
    np.float32(points1), np.float32(points2), F, imgSize=(w1, h1), threshold=0.5,
)

############## Undistort (Rectify) ##############
imgL = cv2.warpPerspective(imgL, H1, (w1, h1))
imgR = cv2.warpPerspective(imgR, H2, (w2, h2))
cv2.imwrite("undistorted_L.png", imgL)
cv2.imwrite("undistorted_R.png", imgR)
# Plot the images
plt.subplot(1, 2, 1)
plt.title('Undistorted 1')
plt.imshow(imgL)
plt.subplot(1, 2, 2)
plt.title('Undistorted 2')
plt.imshow(imgR)
plt.show()

############## Calculate Disparity (Depth Map) ##############
# Setup two stereo matchers to compute disparity maps both for left and right views.
# Parameters from all steps are defined here to make it easier to adjust values.
resolution     = 0.8    # (0, 1.0] the resolution of the new frame comparing to the old one
brightness     = 0      # [-1.0, 1.0] Additional brightness for the final image
contrast       = 0.7     # [0.0, 3.0] Additional contrast for the final image
filterCap      = 63     # [0, 100]
min_disparity=-12
max_disparity=64
num_disp=max_disparity-min_disparity
block_size=7
uniqueness=7
speckle_window_size=1000
speckle_range=200
lmbda = 80000  # [80000, 100000]
sigma = 1.2
height, width = imgL.shape[:2]
left_matcher = cv2.StereoSGBM_create(
    numDisparities = num_disp,
    blockSize = block_size,
    uniquenessRatio = uniqueness,
    # mode = cv2.StereoSGBM_MODE_HH,
    speckleWindowSize = speckle_window_size,
    speckleRange = speckle_range,
    preFilterCap = 63
)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# Setup a disparity filter to deal with stereo-matching errors.
# It will detect inaccurate disparity values and invalidate them, therefore making the disparity map semi-sparse
# Beside the WLS Filter has an excellent performance on edge preserving smoothing technique
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left = left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

# Perform stereo matching to compute disparity maps for both left and right views.
disparityL = left_matcher.compute(imgL, imgR)
disparityR = right_matcher.compute(imgR, imgL)

# Perform post-filtering
imgLb = cv2.copyMakeBorder(imgL, top = 0, bottom = 0, left = np.uint16(num_disp / resolution), right = 0, borderType= cv2.BORDER_CONSTANT, value = [155,155,155])
filteredImg = wls_filter.filter(disparityL, imgLb, None, disparityR)
# filteredImg = disparityL
# Adjust image resolution, brightness, contrast, and perform disparity truncation hack
filteredImg = filteredImg * resolution
filteredImg = filteredImg + (brightness / 100.0)
filteredImg = (filteredImg - 128) * contrast + 128
filteredImg = np.clip(filteredImg, 0, 255)
filteredImg = np.uint8(filteredImg)
filteredImg = cv2.resize(filteredImg, (width, height), interpolation = cv2.INTER_CUBIC) # Disparity truncation hack
filteredImg = filteredImg[0:height, np.uint16(num_disp / resolution):width]
filteredImg = cv2.resize(filteredImg, (width, height), interpolation = cv2.INTER_CUBIC)  # Disparity truncation hack

plt.imshow(filteredImg)
plt.colorbar()
plt.show()