import cv2
from tqdm import tqdm
import numpy as np
import logging
ID1 = 304773591
ID2 = 313325938
MAX_CORNERS = 200
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 30
BLOCK_SIZE = 3
SMOOTH_RADIUS = 50

my_logger = logging.getLogger('MyLogger')

def fixBorder(frame):
    h, w = frame.shape[0],frame.shape[1]
    # Scale the image 10% without moving the center
    T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.1)
    frame = cv2.warpAffine(frame, T, (w, h))#todo: check
    return frame


def get_video_files(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, width, height, fps





def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'reflect')

    '''Fix padding manually'''
    for i in range(radius):
        curve_pad[i] = curve_pad[radius] - curve_pad[i]

    for i in range(len(curve_pad) - 1, len(curve_pad) - 1 - radius, -1):
        curve_pad[i] = curve_pad[len(curve_pad) - radius - 1] - curve_pad[i]

    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed


def smooth(trajectory, smooth_radius):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(smoothed_trajectory.shape[1]):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=smooth_radius)
    return smoothed_trajectory


def write_video(output_path, frames, fps, out_size, is_color):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(output_path, fourcc, fps, out_size, isColor=is_color)
    for frame in frames:
        video_out.write(frame)
    video_out.release()

def load_entire_video(cap, color_space='bgr'):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n_frames):
        success, curr = cap.read()
        if not success:
            break
        if color_space == 'bgr':
            frames.append(curr)
        elif color_space == 'yuv':
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2YUV))
        elif color_space == 'bw':
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY))
        else:
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2HSV))
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.asarray(frames)

def stabilize_video(input_video_path, output_video_path="../Temp/stabilized_{0}_{0}.avi".format(ID1,ID2)):
    my_logger.info('Starting Video Stabilization')
    cap, w, h, fps = get_video_files(path=input_video_path)
    frames_bgr = load_entire_video(cap, color_space='bgr')
    n_frames = len(frames_bgr)
    prev = frames_bgr[0]
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = np.zeros((n_frames - 1, 9), np.float32)
    transforms_list = np.zeros((n_frames - 1, 3, 3), np.float32)


    for  frame_index in tqdm(range(1,n_frames-1), desc='Stabilizing Video Frames number'):

        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=MAX_CORNERS,
                                           qualityLevel=QUALITY_LEVEL,
                                           minDistance=MIN_DISTANCE,
                                           blockSize=BLOCK_SIZE)
        curr_gray = cv2.cvtColor(frames_bgr[frame_index], cv2.COLOR_BGR2GRAY)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)#todo: check

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts, curr_pts = prev_pts[idx], curr_pts[idx]

        # Find transformation matrix
        transform_matrix, _ = cv2.findHomography(prev_pts, curr_pts)#todo: check
        transforms[frame_index] = transform_matrix.flatten()
        prev_gray = curr_gray
        frame_index+=1

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    smoothed_trajectory = smooth(trajectory, SMOOTH_RADIUS)
    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate smooth transformation array
    transforms_smooth = transforms + difference

    stabilized_frames_list = [frames_bgr[0]]
    # Write n_frames-1 transformed frames
    for frame_index, frame in tqdm(enumerate(frames_bgr[:-1]), desc='Applying warps Video Frames number'):
        #print(f'[Video Stabilization] Applying warps to frame: {frame_index+1} / {n_frames - 1}')
        # Apply affine wrapping to the given frame
        #save frame to jpg file
        cv2.imwrite(f'../Temp/frame_{frame_index}.jpg', frame)
        transform_matrix = transforms_smooth[frame_index].reshape((3, 3))
        frame_stabilized = cv2.warpPerspective(frames_bgr[frame_index], transform_matrix, (w, h))
        frame_stabilized = fixBorder(frame_stabilized)
        stabilized_frames_list.append(frame_stabilized)
        cv2.imwrite(f'../Temp/frame_stable{frame_index}.jpg', frame_stabilized)
        cv2.imwrite(f'../Temp/frame_diff{frame_index}.jpg', frame_stabilized-frame)
        transforms_list[frame_index] = transform_matrix

    cap.release()
    cv2.destroyAllWindows()
    write_video(output_video_path, stabilized_frames_list, fps, (w, h), is_color=True)
    transforms_list.dump('../Temp/transforms_video_stab.np')
    print('~~~~~~~~~~~ [Video Stabilization] FINISHED! ~~~~~~~~~~~')
    print('~~~~~~~~~~~ stabilize.avi has been created! ~~~~~~~~~~~')
    my_logger.info('Finished Video Stabilization')

def main():
    stabilize_video('../Temp/INPUT.avi')
if __name__ == '__main__':
    main()
