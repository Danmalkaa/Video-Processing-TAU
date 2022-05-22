import cv2
import numpy as np
import logging
from constants import (
    MAX_CORNERS,
    QUALITY_LEVEL,
    MIN_DISTANCE,
    BLOCK_SIZE,
    SMOOTH_RADIUS,
)

from utils import (
    get_video_files,
    release_video_files,
    smooth,
    fixBorder,
    write_video,
    load_entire_video
)
my_logger = logging.getLogger('MyLogger')


def stabilize_video(input_video_path):
    my_logger.info('Starting Video Stabilization')
    cap, w, h, fps = get_video_files(path=input_video_path)

    frames_bgr = load_entire_video(cap, color_space='bgr')

    n_frames = len(frames_bgr)
    prev = frames_bgr[0]
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = np.zeros((n_frames - 1, 9), np.float32)
    transforms_list = np.zeros((n_frames - 1, 3, 3), np.float32)
    for frame_index, curr in enumerate(frames_bgr[1:]):
        print(f"[Video Stabilization] Collecting transformations from frame: {frame_index+1} / {n_frames - 1}")
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=MAX_CORNERS,
                                           qualityLevel=QUALITY_LEVEL,
                                           minDistance=MIN_DISTANCE,
                                           blockSize=BLOCK_SIZE)

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts, curr_pts = prev_pts[idx], curr_pts[idx]

        # Find transformation matrix
        transform_matrix, _ = cv2.findHomography(prev_pts, curr_pts)

        transforms[frame_index] = transform_matrix.flatten()
        prev_gray = curr_gray

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    smoothed_trajectory = smooth(trajectory, SMOOTH_RADIUS)
    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate smooth transformation array
    transforms_smooth = transforms + difference

    stabilized_frames_list = [frames_bgr[0]]
    # Write n_frames-1 transformed frames
    for frame_index, frame in enumerate(frames_bgr[:-1]):
        print(f'[Video Stabilization] Applying warps to frame: {frame_index+1} / {n_frames - 1}')
        # Apply affine wrapping to the given frame
        transform_matrix = transforms_smooth[frame_index].reshape((3, 3))

        frame_stabilized = cv2.warpPerspective(frame, transform_matrix, (w, h))
        frame_stabilized = fixBorder(frame_stabilized)
        stabilized_frames_list.append(frame_stabilized)
        transforms_list[frame_index] = transform_matrix

    release_video_files(cap)
    write_video('../Outputs/stabilize.avi', stabilized_frames_list, fps, (w, h), is_color=True)
    transforms_list.dump('../Temp/transforms_video_stab.np')
    print('~~~~~~~~~~~ [Video Stabilization] FINISHED! ~~~~~~~~~~~')
    print('~~~~~~~~~~~ stabilize.avi has been created! ~~~~~~~~~~~')
    my_logger.info('Finished Video Stabilization')
