import GeodisTK
import cv2
import numpy as np
import logging
import time
import os
import re

from skimage.graph import MCP
from constants import (
    EPSILON_NARROW_BAND,
    ERODE_ITERATIONS,
    DILATE_ITERATIONS,
    GEODISTK_ITERATIONS,
    KDE_BW,
    R
)
from utils import (
    load_entire_video,
    get_video_files,
    choose_indices_for_foreground,
    choose_indices_for_background,
    write_video,
    fixBorder,
    estimate_pdf
)

my_logger = logging.getLogger('MyLogger')

def reaf_all_frame_from_folser(path_of_folder,typeofimg=".jpg"):
    frame_array = []
    #get array of all files name in the folder
    list_of_files = os.listdir(path_of_folder)
    list_of_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    for file in list_of_files:

        if file.endswith(typeofimg):
            frame = cv2.imread(os.path.join(path_of_folder, file))
            frame_array.append(frame)
    return frame_array


def video_matting(input_stabilize_video, binary_video_path, new_background):
    start_video_time = time.time()
    time_list_1,time_list_2,time_list_3,time_list_4,time_list_5,time_list_6,time_list_7 = [],[],[],[],[],[],[]
    my_logger.info('Starting Matting')

    # Read input video
    frames_bgr = np.array(reaf_all_frame_from_folser('../Temp/stabilized_frames/'))
    w, h = frames_bgr[0].shape[0:2][::-1]
    fps_stabilize = 30.04
    # cap_stabilize, w, h, fps_stabilize = get_video_files(path=input_stabilize_video)
    cap_binary, _, _, fps_binary = get_video_files(path=binary_video_path)

    # Get frame count
    n_frames = frames_bgr.shape[0]
    # n_frames = int(cap_stabilize.get(cv2.CAP_PROP_FRAME_COUNT))

    # frames_bgr = load_entire_video(cap_stabilize, color_space='bgr')
    frames_yuv = []
    for curr in frames_bgr:
        frame_yuv = cv2.cvtColor(curr, cv2.COLOR_BGR2YUV)
        frames_yuv.append(frame_yuv)
    frames_yuv = np.array(frames_yuv)
    # frames_yuv = load_entire_video(cap_stabilize, color_space='yuv')
    frames_binary = load_entire_video(cap_binary, color_space='bw')

    '''Resize new background'''
    new_background = cv2.resize(new_background, (w, h))

    '''Starting Matting Process'''
    full_matted_frames_list, alpha_frames_list = [], []
    for frame_index in range(n_frames):
        print(f'[Matting] - Frame: {frame_index} / {n_frames}')
        luma_frame, _, _ = cv2.split(frames_yuv[frame_index])
        bgr_frame = frames_bgr[frame_index]

        # Check non_zero frame
        if len(np.nonzero(bgr_frame[0])[0]) == 0:
            full_matted_frames_list.append(frames_bgr[frame_index])
            alpha_frames_list.append(frames_binary[frame_index])
            continue

        original_mask_frame = frames_binary[frame_index]
        original_mask_frame = (original_mask_frame > 150).astype(np.uint8)

        '''Find indices for resizing image to work only on relevant part!'''
        start_time = time.time()
        DELTA = 20
        binary_frame_rectangle_x_axis = np.where(original_mask_frame == 1)[1]
        left_index, right_index = np.min(binary_frame_rectangle_x_axis), np.max(binary_frame_rectangle_x_axis)
        left_index, right_index = max(0, left_index - DELTA), min(right_index + DELTA, original_mask_frame.shape[1] - 1)
        binary_frame_rectangle_y_axis = np.where(original_mask_frame == 1)[0]
        top_index, bottom_index = np.min(binary_frame_rectangle_y_axis), np.max(binary_frame_rectangle_y_axis)
        top_index, bottom_index = max(0, top_index - DELTA), min(bottom_index + DELTA, original_mask_frame.shape[0] - 1)
        end_time = time.time()
        time_list_1.append(end_time-start_time)

        ''' Resize images '''
        smaller_luma_frame = luma_frame[top_index:bottom_index, left_index:right_index]
        smaller_bgr_frame = bgr_frame[top_index:bottom_index, left_index:right_index]
        smaller_new_background = new_background[top_index:bottom_index, left_index:right_index]

        '''Erode & Resize foreground mask & Build distance map for foreground'''
        foreground_mask = cv2.erode(original_mask_frame, np.ones((3, 3)), iterations=ERODE_ITERATIONS)
        smaller_foreground_mask = foreground_mask[top_index:bottom_index, left_index:right_index]
        start_time = time.time()
        smaller_foreground_distance_map = GeodisTK.geodesic2d_raster_scan(smaller_luma_frame, smaller_foreground_mask,
                                                                          1.0, GEODISTK_ITERATIONS)

        end_time = time.time()
        time_list_2.append(end_time-start_time)

        '''Dilate & Resize image & Build distance map for background'''
        background_mask = cv2.dilate(original_mask_frame, np.ones((3, 3)), iterations=DILATE_ITERATIONS)
        background_mask = 1 - background_mask
        smaller_background_mask = background_mask[top_index:bottom_index, left_index:right_index]
        start_time = time.time()
        smaller_background_distance_map = GeodisTK.geodesic2d_raster_scan(smaller_luma_frame, smaller_background_mask,
                                                                          1.0, GEODISTK_ITERATIONS)
        end_time = time.time()
        time_list_3.append(end_time-start_time)

        ''' Building narrow band undecided zone'''
        start_time = time.time()
        smaller_foreground_distance_map = smaller_foreground_distance_map / (smaller_foreground_distance_map + smaller_background_distance_map)
        smaller_background_distance_map = 1 - smaller_foreground_distance_map
        smaller_narrow_band_mask = (np.abs(smaller_foreground_distance_map - smaller_background_distance_map) < EPSILON_NARROW_BAND).astype(np.uint8)
        smaller_narrow_band_mask_indices = np.where(smaller_narrow_band_mask == 1)

        smaller_decided_foreground_mask = (smaller_foreground_distance_map < smaller_background_distance_map - EPSILON_NARROW_BAND).astype(np.uint8)
        smaller_decided_background_mask = (smaller_background_distance_map >= smaller_foreground_distance_map - EPSILON_NARROW_BAND).astype(np.uint8)
        end_time = time.time()
        time_list_4.append(end_time-start_time)

        '''Building KDEs for foreground & background to calculate priors for alpha calculation'''
        start_time = time.time()
        omega_f_indices = choose_indices_for_foreground(smaller_decided_foreground_mask, 200)
        omega_b_indices = choose_indices_for_background(smaller_decided_background_mask, 200)
        foreground_pdf = estimate_pdf(original_frame=smaller_bgr_frame, indices=omega_f_indices, bw_method=KDE_BW)
        background_pdf = estimate_pdf(original_frame=smaller_bgr_frame, indices=omega_b_indices, bw_method=KDE_BW)
        smaller_narrow_band_foreground_probs = foreground_pdf(smaller_bgr_frame[smaller_narrow_band_mask_indices])
        smaller_narrow_band_background_probs = background_pdf(smaller_bgr_frame[smaller_narrow_band_mask_indices])
        end_time = time.time()
        time_list_5.append(end_time-start_time)

        '''Start creating alpha map'''
        start_time = time.time()
        w_f = np.power(smaller_foreground_distance_map[smaller_narrow_band_mask_indices],-R) * smaller_narrow_band_foreground_probs
        w_b = np.power(smaller_background_distance_map[smaller_narrow_band_mask_indices],-R) * smaller_narrow_band_background_probs
        alpha_narrow_band = w_f / (w_f + w_b)
        smaller_alpha = np.copy(smaller_decided_foreground_mask).astype(np.float)
        smaller_alpha[smaller_narrow_band_mask_indices] = alpha_narrow_band
        end_time = time.time()
        time_list_6.append(end_time-start_time)

        '''Naive implementation for matting as described in algorithm'''
        smaller_matted_frame = smaller_alpha[:, :, np.newaxis] * smaller_bgr_frame + (1 - smaller_alpha[:, :, np.newaxis]) * smaller_new_background

        '''move from small rectangle to original size'''
        start_time = time.time()
        full_matted_frame = np.copy(new_background)
        full_matted_frame[top_index:bottom_index, left_index:right_index] = smaller_matted_frame
        full_matted_frames_list.append(full_matted_frame)

        full_alpha_frame = np.zeros(original_mask_frame.shape)
        full_alpha_frame[top_index:bottom_index, left_index:right_index] = smaller_alpha
        full_alpha_frame = (full_alpha_frame * 255).astype(np.uint8)
        alpha_frames_list.append(full_alpha_frame)
        end_time = time.time()
        time_list_7.append(end_time-start_time)

    # create_unstabilized_alpha(alpha_frames_list=alpha_frames_list, fps=fps_stabilize)
    write_video(output_path='../Outputs/matted.avi', frames=full_matted_frames_list, fps=fps_stabilize, out_size=(w, h),
                is_color=True)
    write_video(output_path='../Outputs/alpha.avi', frames=alpha_frames_list, fps=fps_stabilize, out_size=(w, h), is_color=False)
    print('~~~~~~~~~~~ [Matting] FINISHED! ~~~~~~~~~~~')
    print('~~~~~~~~~~~ matted.avi has been created! ~~~~~~~~~~~')
    print('~~~~~~~~~~~ alpha.avi has been created! ~~~~~~~~~~~')
    for lis in [time_list_1,time_list_2,time_list_3,time_list_4,time_list_5,time_list_6,time_list_7]:
        print(np.mean(np.array(lis)))
    # print('~~~~~~~~~~~ unstabilized_alpha.avi has been created! ~~~~~~~~~~~')
    end_video_time = time.time()
    print(-1*(start_video_time-end_video_time))
    my_logger.info('Finished Matting')


# def create_unstabilized_alpha(alpha_frames_list, fps):
#     transforms_list = np.load('../Temp/transforms_video_stab.np', allow_pickle=True)
#     unstabilized_frames_list = [alpha_frames_list[0]]
#     h, w = alpha_frames_list[0].shape
#     for i in range(len(alpha_frames_list) - 1):
#         m = np.linalg.inv(transforms_list[i])
#         unstabilized_frame = cv2.warpPerspective(alpha_frames_list[i + 1], m, (w, h))
#         unstabilized_frame = fixBorder(unstabilized_frame)
#         unstabilized_frames_list.append(unstabilized_frame)
#
#     write_video('../Outputs/unstabilized_alpha.avi', frames=unstabilized_frames_list, fps=fps, out_size=(w, h), is_color=False)


video_matting('../Temp/304773591_313325938_faster_stabilized_video.avi', '../Temp/binary.avi', cv2.imread('../Inputs/background.jpg'))
