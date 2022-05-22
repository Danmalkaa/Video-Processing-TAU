import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
input_video_path = '../Temp/INPUT.avi'

def read_frame(input_video_path):
    frame_list = []

    filename = input_video_path
    # Read the video file
    cap = cv2.VideoCapture(filename)
    # Get the first frame
    ret, f1= cap.read()
    # Get the secent frame
    ret, f2 = cap.read()


    #covert to grayscale
    f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    C = np.dstack((f2,f1,f2))
    # cv2.imshow("imfuse", C)
    # cv2.waitKey(0)
    return f1, f2
def collect_salient_points(input_video_path):
    ptThresh = 0.1
    f1,f2=read_frame(input_video_path)


    corners_1 = cv2.goodFeaturesToTrack(f1, 50, 0.1, 20)
    corners_1 = np.int0(corners_1)
    corners_2 = cv2.goodFeaturesToTrack(f2, 50, 0.1, 20)
    corners_2 = np.int0(corners_2)
    cor_diff= corners_1 - corners_2
    #cor_diff_mask=  np.abs(corners_1 - corners_2)<(10,10)
    cor_diff_mask=  np.abs(cor_diff)<(10,10)

    cor_diff_new=cor_diff*cor_diff_mask
    return corners_1, corners_2
def extract_freak_descriptors_for_the_corners(input_video_path):
    """
    Extracts FREAK descriptors for the corners.
    """
    img1,img2=read_frame(input_video_path)   #read the first and second frame
    corner1, corner2 = collect_salient_points(input_video_path)

    freak = cv2.xfeatures2d.FREAK_create()
    kp1, des1 = freak.compute(img1,corner1)
    kp2, des2 = freak.compute(img2,corner2)
    #show the keypoints
    img1 = cv2.drawKeypoints(corner1, kp1, None, color=(0, 255, 0))
    img2 = cv2.drawKeypoints(corner2, kp2, None, color=(0, 255, 0))
    cv2.imshow(img1)
    cv2.waitKey(0)
    return des1, des2
def Extract_FREAK_descriptors_for_the_corners(input_video_path):
    """
    Extracts FREAK descriptors for the corners.
    :param corners_1:
    :param corners_2:
    :return:
    """
    img1,img2=read_frame(input_video_path)
    # img1 = cv2.imread('box.png', cv.IMREAD_GRAYSCALE)  # queryImage
    # img2 = cv2.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE)  # trainImage
    # Initiate SIFT detector
    sift = cv2.SIFT_create(nfeatures=200)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=20)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()

def Estimating_Transform_from_Noisy_Correspondences(input_video_path):

    corners_1, corners_2 = collect_salient_points(input_video_path)



def main():
    Extract_FREAK_descriptors_for_the_corners(input_video_path)
if __name__ == '__main__':
    main()