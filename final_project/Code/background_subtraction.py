import logging
import cv2
from tqdm import tqdm
import numpy as np
import time
from scipy.stats import gaussian_kde
ID1 = 304773591
ID2 = 313325938
from constants import (
    BW_MEDIUM,
    SHOES_HEIGHT,
    SHOULDERS_HEIGHT,
    LEGS_HEIGHT,
    BW_NARROW,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    BLUE_MASK_THR,
    FACE_WINDOW_HEIGHT,
    FACE_WINDOW_WIDTH
)
from utils import (
    get_video_files,
    load_entire_video,
    apply_mask_on_color_frame,
    write_video,
    release_video_files,
    disk_kernel,
    scale_matrix_0_to_255
)
def check_in_dict(dict, element, function):
    if element in dict:
        return dict[element]
    else:
        dict[element] = function(np.asarray(element))[0]
        return dict[element]

def new_estimate_pdf(omega_values, bw_method):
    pdf = gaussian_kde(omega_values.T, bw_method=bw_method)
    return lambda x: pdf(x.T)
def choosIndicesForBackgroundAndForeground(mask, number_of_choices,b_or_f):
    indices = np.where(mask == b_or_f)
    if len(indices[0]) == 0:
        return np.column_stack((indices[0],indices[1]))
    indices_choices = np.random.choice(len(indices[0]), number_of_choices)
    return np.column_stack((indices[0][indices_choices], indices[1][indices_choices]))


def backgroundSubTractorKNNStudyingFrameshistory(frames_hsv,mask_list,backSub):
    for j in tqdm( range(6), desc="BackgroundSubtractorKNN Studying Frames history"):
        for index_frame, frame in enumerate(frames_hsv):
            frame_sv = frame[:, :, 1:]
            fg_Mask = backSub.apply(frame_sv)
            fg_Mask = (fg_Mask > 200).astype(np.uint8)
            mask_list[index_frame] = fg_Mask
            #save frame as img
            name= "../ScreenRec/frame_{0}".format(index_frame)+"_"+str(j)+".jpg"
            cv2.imwrite(name, fg_Mask * 255)
def collectingColorsBodyAndShoes(mask_list,frames_bgr,person_and_blue_mask_list,omega_f_colors,omega_b_colors,omega_f_shoes_colors,omega_b_shoes_colors):
    for frame_index, frame in enumerate(frames_bgr):
        blue_frame, _, _ = cv2.split(frame)
        mask_for_frame = mask_list[frame_index].astype(np.uint8)
        mask_for_frame = cv2.morphologyEx(mask_for_frame, cv2.MORPH_CLOSE, disk_kernel(6))#todo: check morphologyEx
        mask_for_frame = cv2.medianBlur(mask_for_frame, 7)#todo: check medianBlur
        contours, _ = cv2.findContours(mask_for_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)
        person_mask = np.zeros(mask_for_frame.shape)
        cv2.fillPoly(person_mask, pts=[contours[0]], color=1)#todo: check fillPoly
        blue_mask = (blue_frame < BLUE_MASK_THR).astype(np.uint8)
        person_and_blue_mask = (person_mask * blue_mask).astype(np.uint8)
        omega_f_indices = choosIndicesForBackgroundAndForeground(person_and_blue_mask, 20,1)
        omega_b_indices = choosIndicesForBackgroundAndForeground(person_and_blue_mask, 20,0)
        shoes_mask = np.copy(person_and_blue_mask)
        shoes_mask[:SHOES_HEIGHT, :] = 0
        omega_f_shoes_indices = choosIndicesForBackgroundAndForeground(shoes_mask, 20,1)
        shoes_mask = np.copy(person_and_blue_mask)
        shoes_mask[:SHOES_HEIGHT - 120, :] = 1
        omega_b_shoes_indices = choosIndicesForBackgroundAndForeground(shoes_mask, 20,0)
        person_and_blue_mask_list[frame_index] = person_and_blue_mask
        if omega_f_colors is None:
            omega_f_colors = frame[omega_f_indices[:, 0], omega_f_indices[:, 1], :]
            omega_b_colors = frame[omega_b_indices[:, 0], omega_b_indices[:, 1], :]
            omega_f_shoes_colors = frame[omega_f_shoes_indices[:, 0], omega_f_shoes_indices[:, 1], :]
            omega_b_shoes_colors = frame[omega_b_shoes_indices[:, 0], omega_b_shoes_indices[:, 1], :]
        else:
            omega_f_colors = np.concatenate((omega_f_colors, frame[omega_f_indices[:, 0], omega_f_indices[:, 1], :]))
            omega_b_colors = np.concatenate((omega_b_colors, frame[omega_b_indices[:, 0], omega_b_indices[:, 1], :]))
            omega_f_shoes_colors = np.concatenate(
                (omega_f_shoes_colors, frame[omega_f_shoes_indices[:, 0], omega_f_shoes_indices[:, 1], :]))
            omega_b_shoes_colors = np.concatenate(
                (omega_b_shoes_colors, frame[omega_b_shoes_indices[:, 0], omega_b_shoes_indices[:, 1], :]))
    return mask_list,frames_bgr,person_and_blue_mask_list,omega_f_colors,omega_b_colors,omega_f_shoes_colors,omega_b_shoes_colors
def FilteringKDEs(omega_f_colors,omega_b_colors,omega_f_shoes_colors,omega_b_shoes_colors,n_frames,h, w,frames_bgr,person_and_blue_mask_list):
    foreground_pdf = new_estimate_pdf(omega_values=omega_f_colors, bw_method=BW_MEDIUM)
    background_pdf = new_estimate_pdf(omega_values=omega_b_colors, bw_method=BW_MEDIUM)
    foreground_shoes_pdf = new_estimate_pdf(omega_values=omega_f_shoes_colors, bw_method=BW_MEDIUM)
    background_shoes_pdf = new_estimate_pdf(omega_values=omega_b_shoes_colors, bw_method=BW_MEDIUM)

    foreground_pdf_memoization, background_pdf_memoization = dict(), dict()
    foreground_shoes_pdf_memoization, background_shoes_pdf_memoization = dict(), dict()
    or_mask_list = np.zeros((n_frames, h, w))
    for frame_index, frame in enumerate(frames_bgr):
        person_and_blue_mask = person_and_blue_mask_list[frame_index]
        person_and_blue_mask_indices = np.where(person_and_blue_mask == 1)
        y_mean, x_mean = int(np.mean(person_and_blue_mask_indices[0])), int(np.mean(person_and_blue_mask_indices[1]))
        small_frame_bgr = frame[max(0, y_mean - WINDOW_HEIGHT // 2):min(h, y_mean + WINDOW_HEIGHT // 2),
                          max(0, x_mean - WINDOW_WIDTH // 2):min(w, x_mean + WINDOW_WIDTH // 2),
                          :]
        small_person_and_blue_mask = person_and_blue_mask[
                                     max(0, y_mean - WINDOW_HEIGHT // 2):min(h, y_mean + WINDOW_HEIGHT // 2),
                                     max(0, x_mean - WINDOW_WIDTH // 2):min(w, x_mean + WINDOW_WIDTH // 2)]

        small_person_and_blue_mask_indices = np.where(small_person_and_blue_mask == 1)
        small_probs_fg_bigger_bg_mask = np.zeros(small_person_and_blue_mask.shape)
        small_foreground_probabilities_stacked = np.fromiter(
            map(lambda elem: check_in_dict(foreground_pdf_memoization, elem, foreground_pdf),
                map(tuple, small_frame_bgr[small_person_and_blue_mask_indices])), dtype=float)
        small_background_probabilities_stacked = np.fromiter(
            map(lambda elem: check_in_dict(background_pdf_memoization, elem, background_pdf),
                map(tuple, small_frame_bgr[small_person_and_blue_mask_indices])), dtype=float)
        small_probs_fg_bigger_bg_mask[small_person_and_blue_mask_indices] = (
                small_foreground_probabilities_stacked > small_background_probabilities_stacked).astype(np.uint8)

        '''Shoes restoration'''
        smaller_upper_white_mask = np.copy(small_probs_fg_bigger_bg_mask)
        smaller_upper_white_mask[:-270, :] = 1
        small_probs_fg_bigger_bg_mask_black_indices = np.where(smaller_upper_white_mask == 0)
        small_probs_shoes_fg_bigger_bg_mask = np.zeros(small_person_and_blue_mask.shape)
        small_shoes_foreground_probabilities_stacked = np.fromiter(
            map(lambda elem: check_in_dict(foreground_shoes_pdf_memoization, elem, foreground_shoes_pdf),
                map(tuple, small_frame_bgr[small_probs_fg_bigger_bg_mask_black_indices])), dtype=float)

        small_shoes_background_probabilities_stacked = np.fromiter(
            map(lambda elem: check_in_dict(background_shoes_pdf_memoization, elem, background_shoes_pdf),
                map(tuple, small_frame_bgr[small_probs_fg_bigger_bg_mask_black_indices])), dtype=float)

        shoes_fg_shoes_bg_ratio = small_shoes_foreground_probabilities_stacked / (
                small_shoes_foreground_probabilities_stacked + small_shoes_background_probabilities_stacked)
        shoes_fg_beats_shoes_bg_mask = (shoes_fg_shoes_bg_ratio > 0.75).astype(np.uint8)
        small_probs_shoes_fg_bigger_bg_mask[small_probs_fg_bigger_bg_mask_black_indices] = shoes_fg_beats_shoes_bg_mask
        small_probs_shoes_fg_bigger_bg_mask_indices = np.where(small_probs_shoes_fg_bigger_bg_mask == 1)
        if len(small_probs_shoes_fg_bigger_bg_mask_indices[0]) > 0:
            y_shoes_mean, x_shoes_mean = int(np.mean(small_probs_shoes_fg_bigger_bg_mask_indices[0])), int( np.mean(small_probs_shoes_fg_bigger_bg_mask_indices[1]))
        else:
            y_shoes_mean, x_shoes_mean=0,0
        small_or_mask = np.zeros(small_probs_fg_bigger_bg_mask.shape)
        small_or_mask[:y_shoes_mean, :] = small_probs_fg_bigger_bg_mask[:y_shoes_mean, :]
        small_or_mask[y_shoes_mean:, :] = np.maximum(small_probs_fg_bigger_bg_mask[y_shoes_mean:, :], small_probs_shoes_fg_bigger_bg_mask[y_shoes_mean:, :]).astype(            np.uint8)

        DELTA_Y = 30
        small_or_mask[y_shoes_mean - DELTA_Y:, :] = cv2.morphologyEx(small_or_mask[y_shoes_mean - DELTA_Y:, :],
                                                                     cv2.MORPH_CLOSE, np.ones((1, 20)))
        small_or_mask[y_shoes_mean - DELTA_Y:, :] = cv2.morphologyEx(small_or_mask[y_shoes_mean - DELTA_Y:, :],
                                                                     cv2.MORPH_CLOSE, disk_kernel(20))

        or_mask = np.zeros(person_and_blue_mask.shape)
        or_mask[max(0, y_mean - WINDOW_HEIGHT // 2):min(h, y_mean + WINDOW_HEIGHT // 2),
        max(0, x_mean - WINDOW_WIDTH // 2):min(w, x_mean + WINDOW_WIDTH // 2)] = small_or_mask
        or_mask_list[frame_index] = or_mask
    return or_mask_list
def collectinColorsFace(frames_bgr,or_mask_list,h,w):
    omega_f_face_colors, omega_b_face_colors = None, None
    for frame_index, frame in enumerate(frames_bgr):
        or_mask = or_mask_list[frame_index]
        face_mask = np.copy(or_mask)
        face_mask[SHOULDERS_HEIGHT:, :] = 0
        face_mask_indices = np.where(face_mask == 1)
        y_mean, x_mean = int(np.mean(face_mask_indices[0])), int(np.mean(face_mask_indices[1]))

        small_frame_bgr = frame[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):min(h, y_mean + FACE_WINDOW_HEIGHT // 2),
                          max(0, x_mean - FACE_WINDOW_WIDTH // 2):min(w, x_mean + FACE_WINDOW_WIDTH // 2),
                          :]
        face_mask = np.copy(or_mask)  # Restore face mask again
        small_face_mask = face_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):min(h, y_mean + FACE_WINDOW_HEIGHT // 2),
                          max(0, x_mean - FACE_WINDOW_WIDTH // 2):min(w, x_mean + FACE_WINDOW_WIDTH // 2)]

        small_face_mask = cv2.morphologyEx(small_face_mask, cv2.MORPH_OPEN, np.ones((20, 1), np.uint8))#todo: check morphologyEx
        small_face_mask = cv2.morphologyEx(small_face_mask, cv2.MORPH_OPEN, np.ones((1, 20), np.uint8))

        omega_f_face_indices = choosIndicesForBackgroundAndForeground(small_face_mask, 20,1)
        omega_b_face_indices = choosIndicesForBackgroundAndForeground(small_face_mask, 20,0)

        if omega_f_face_colors is None:
            omega_f_face_colors = small_frame_bgr[omega_f_face_indices[:, 0], omega_f_face_indices[:, 1], :]
            omega_b_face_colors = small_frame_bgr[omega_b_face_indices[:, 0], omega_b_face_indices[:, 1], :]
        else:
            omega_f_face_colors = np.concatenate(
                (omega_f_face_colors, small_frame_bgr[omega_f_face_indices[:, 0], omega_f_face_indices[:, 1], :]))
            omega_b_face_colors = np.concatenate(
                (omega_b_face_colors, small_frame_bgr[omega_b_face_indices[:, 0], omega_b_face_indices[:, 1], :]))
    return omega_f_face_colors, omega_b_face_colors
def finalprossec(omega_f_face_colors,omega_b_face_colors,frames_bgr,or_mask_list,h,w):
    foreground_face_pdf = new_estimate_pdf(omega_values=omega_f_face_colors, bw_method=BW_NARROW)
    background_face_pdf = new_estimate_pdf(omega_values=omega_b_face_colors, bw_method=BW_NARROW)
    foreground_face_pdf_memoization, background_face_pdf_memoization = dict(), dict()
    final_masks_list, final_frames_list = [], []
    for frame_index, frame in enumerate(frames_bgr):
        or_mask = or_mask_list[frame_index]
        face_mask = np.copy(or_mask)
        face_mask[SHOULDERS_HEIGHT:, :] = 0
        face_mask_indices = np.where(face_mask == 1)
        y_mean, x_mean = int(np.mean(face_mask_indices[0])), int(np.mean(face_mask_indices[1]))

        small_frame_bgr = frame[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):min(h, y_mean + FACE_WINDOW_HEIGHT // 2),
                          max(0, x_mean - FACE_WINDOW_WIDTH // 2):min(w, x_mean + FACE_WINDOW_WIDTH // 2),
                          :]
        face_mask = np.copy(or_mask)  # Restore face mask again
        small_face_mask = face_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):min(h, y_mean + FACE_WINDOW_HEIGHT // 2),
                          max(0, x_mean - FACE_WINDOW_WIDTH // 2):min(w, x_mean + FACE_WINDOW_WIDTH // 2)]

        small_frame_bgr_stacked = small_frame_bgr.reshape((-1, 3))

        small_face_foreground_probabilities_stacked = np.fromiter(
            map(lambda elem: check_in_dict(foreground_face_pdf_memoization, elem, foreground_face_pdf),
                map(tuple, small_frame_bgr_stacked)), dtype=float)
        small_face_background_probabilities_stacked = np.fromiter(
            map(lambda elem: check_in_dict(background_face_pdf_memoization, elem, background_face_pdf),
                map(tuple, small_frame_bgr_stacked)), dtype=float)

        small_face_foreground_probabilities = small_face_foreground_probabilities_stacked.reshape(small_face_mask.shape)
        small_face_background_probabilities = small_face_background_probabilities_stacked.reshape(small_face_mask.shape)
        small_probs_face_fg_bigger_face_bg_mask = (
                small_face_foreground_probabilities > small_face_background_probabilities).astype(np.uint8)
        small_probs_face_fg_bigger_face_bg_mask_laplacian = cv2.Laplacian(small_probs_face_fg_bigger_face_bg_mask,
                                                                          cv2.CV_32F)
        small_probs_face_fg_bigger_face_bg_mask_laplacian = np.abs(small_probs_face_fg_bigger_face_bg_mask_laplacian)
        small_probs_face_fg_bigger_face_bg_mask = np.maximum(
            small_probs_face_fg_bigger_face_bg_mask - small_probs_face_fg_bigger_face_bg_mask_laplacian, 0)
        small_probs_face_fg_bigger_face_bg_mask[np.where(small_probs_face_fg_bigger_face_bg_mask > 1)] = 0
        small_probs_face_fg_bigger_face_bg_mask = small_probs_face_fg_bigger_face_bg_mask.astype(np.uint8)

        contours, _ = cv2.findContours(small_probs_face_fg_bigger_face_bg_mask, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)
        small_contour_mask = np.zeros(small_probs_face_fg_bigger_face_bg_mask.shape, dtype=np.uint8)
        cv2.fillPoly(small_contour_mask, pts=[contours[0]], color=1)

        small_contour_mask = cv2.morphologyEx(small_contour_mask, cv2.MORPH_CLOSE, disk_kernel(12))
        small_contour_mask = cv2.dilate(small_contour_mask, disk_kernel(3), iterations=1).astype(np.uint8)
        small_contour_mask[-50:, :] = small_face_mask[-50:, :]

        final_mask = np.copy(or_mask).astype(np.uint8)
        final_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):min(h, y_mean + FACE_WINDOW_HEIGHT // 2),
        max(0, x_mean - FACE_WINDOW_WIDTH // 2):min(w, x_mean + FACE_WINDOW_WIDTH // 2)] = small_contour_mask
        final_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):LEGS_HEIGHT, :] = cv2.morphologyEx(
            final_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):LEGS_HEIGHT, :], cv2.MORPH_OPEN,
            np.ones((6, 1), np.uint8))
        final_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):LEGS_HEIGHT, :] = cv2.morphologyEx(
            final_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):LEGS_HEIGHT, :], cv2.MORPH_OPEN,
            np.ones((1, 6), np.uint8))

        contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)
        final_contour_mask = np.zeros(final_mask.shape)
        cv2.fillPoly(final_contour_mask, pts=[contours[0]], color=1)
        final_mask = (final_contour_mask * final_mask).astype(np.uint8)
        final_masks_list.append(scale_matrix_0_to_255(final_mask))
        final_frames_list.append(apply_mask_on_color_frame(frame=frame, mask=final_mask))
    return final_masks_list, final_frames_list
def background_subtraction(input_video_path, output_video_path="../Temp/extracted_{0}_{0}.avi".format(ID1,ID2)):
    start_time= time.time()
    # Read input video
    cap, w, h, fps = get_video_files(path=input_video_path)
    # Get frame count
    frames_bgr = load_entire_video(cap, color_space='bgr')
    frames_hsv = load_entire_video(cap, color_space='hsv')
    n_frames = len(frames_bgr)
    backSub = cv2.createBackgroundSubtractorKNN()
    mask_list = np.zeros((n_frames, h, w)).astype(np.uint8)
    print(f"[BS] - BackgroundSubtractorKNN Studying Frames history")
    backgroundSubTractorKNNStudyingFrameshistory(frames_hsv,mask_list,backSub)
    print(f"[BS] - BackgroundSubtractorKNN Finished the time is: {time.time() - start_time}")

    omega_f_colors, omega_b_colors = None, None
    omega_f_shoes_colors, omega_b_shoes_colors = None, None
    person_and_blue_mask_list = np.zeros((n_frames, h, w))

    start_time_collecting_colors= time.time()
    print("start Collecting colors for building body & shoes KDEs")
    mask_list,frames_bgr,person_and_blue_mask_list,omega_f_colors,omega_b_colors,omega_f_shoes_colors,omega_b_shoes_colors=collectingColorsBodyAndShoes(mask_list,frames_bgr,person_and_blue_mask_list,omega_f_colors,omega_b_colors,omega_f_shoes_colors,omega_b_shoes_colors)
    print(f"end  Collecting colors for building body & shoes KDEs the time is: {time.time() - start_time_collecting_colors}")




    print("start Filtering with KDEs general body parts & shoes")
    start_time_filtering_general_body_parts_shoes = time.time()
    or_mask_list=FilteringKDEs(omega_f_colors, omega_b_colors, omega_f_shoes_colors, omega_b_shoes_colors, n_frames, h, w,
                  frames_bgr, person_and_blue_mask_list)
    print(f"end  Filtering with KDEs general body parts & shoes the time is: {time.time() - start_time_filtering_general_body_parts_shoes}")

    print("start Collecting colors for face")
    start_time_collecting_colors_face = time.time()
    omega_f_face_colors, omega_b_face_colors=collectinColorsFace(frames_bgr,or_mask_list,h,w)
    print(f"end  Collecting colors for face the time is: {time.time() - start_time_collecting_colors}")

    print("start Final Processing")
    start_time_final_processing = time.time()
    final_masks_list, final_frames_list =finalprossec(omega_f_face_colors,omega_b_face_colors,frames_bgr,or_mask_list,h,w)
    print(f"end  Final Processing the time is: {time.time() - start_time_final_processing}")

    write_video(output_path=output_video_path, frames=final_frames_list, fps=fps, out_size=(w, h), is_color=True)
    write_video(output_path='../Temp/binary.avi', frames=final_masks_list, fps=fps, out_size=(w, h), is_color=False)
    print('~~~~~~~~~~~ [BS] FINISHED! ~~~~~~~~~~~')
    print('~~~~~~~~~~~ binary.avi has been created! ~~~~~~~~~~~')
    print('~~~~~~~~~~~ extracted.avi has been created! ~~~~~~~~~~~')
    print("the time of all is:" + str(time.time() - start_time))

    release_video_files(cap)
def main():
    background_subtraction("../Temp/stabilized_{0}_{0}.avi".format(ID1,ID2))
if __name__ == '__main__':
    main()