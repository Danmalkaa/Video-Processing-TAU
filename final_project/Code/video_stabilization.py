import logging
import cv2
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import griddata
import time

# ID1 = 304773591
# ID2 = 313325938
# MAX_CORNERS = 200
# QUALITY_LEVEL = 0.01
# MIN_DISTANCE = 30
# BLOCK_SIZE = 3
# SMOOTH_RADIUS = 50

# FILL IN YOUR ID
ID1 = 304773591
ID2 = 313325938

# Choose parameters
# WINDOW_SIZE_TAU = 5  # Add your value here!
# MAX_ITER_TAU = 4  # Add your value here!
# NUM_LEVELS_TAU = 5  # Add your value here!
WINDOW_SIZE_TAU = 5  # Add your value here!
MAX_ITER_TAU = 5  # Add your value here!
NUM_LEVELS_TAU = 6  # Add your value here!

my_logger = logging.getLogger('MyLogger')





# FILL IN YOUR ID
ID1 = 304773591
ID2 = 313325938

PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

WINDOW_SIZE = 5
def get_video_files(path, output_name, isColor):
    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define video codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (width, height)
    out = cv2.VideoWriter(output_name, fourcc, fps, out_size, isColor=isColor)
    return cap, out
def read_frame_as_a_jpg_file_to_array(n):
    """
    this function reads the frame n from the jpg file and returns it as a  array
    this function is used us to debug the code and use finelly
    :param n: the number of the frame to read
    :return: lst: the list of the frame
    """
    lst = []
    for i in range(n):
        img = cv2.imread("frame_f%d.jpg" % i)
        lst.append(img)
    return lst
def array_of_frame_to_avi_file(array_of_frames, output_video_path,fps):
    """
    this function takes an array of frames and save it to an avi file
    :param array_of_frames:
    :param output_video_path:
    :return: None
    """
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size=(array_of_frames[0].shape[1],array_of_frames[0].shape[0])
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    for i in range(len(array_of_frames)):
        #resize the frame
        array_of_frames[i] = cv2.resize(np.uint8(array_of_frames[i]), frame_size)
        if len(array_of_frames[i].shape)==2:
            array_of_frames[i] = cv2.cvtColor(array_of_frames[i], cv2.COLOR_GRAY2RGB)
        # write the flipped frame
        out.write(array_of_frames[i])
    out.release()
    return None

def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.
    Args:
        capture: cv2.VideoCapture object.
    Returns:
        parameters: dict. Video parameters extracted from the video.
    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}

def build_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    """Coverts image to a pyramid list of size num_levels.
    First, create a list with the original image in it. Then, iterate over the
    levels. In each level, convolve the PYRAMID_FILTER with the image from the
    previous level. Then, decimate the result using indexing: simply pick
    every second entry of the result.
    Hint: Use signal.convolve2d with boundary='symm' and mode='same'.
    Args:
        image: np.ndarray. Input image.
        num_levels: int. The number of blurring / decimation times.
    Returns:
        pyramid: list. A list of np.ndarray of images.
    Note that the list length should be num_levels + 1 as the in first entry of
    the pyramid is the original image.
    You are not allowed to use cv2 PyrDown here (or any other cv2 method).
    We use a slightly different decimation process from this function.
    """
    pyramid = [image.copy()]
    """INSERT YOUR CODE HERE."""  # TODO: Check it works with the list comprehension
    for i in range(num_levels):
        pyramid.append(signal.convolve2d(pyramid[i], PYRAMID_FILTER, mode='same', boundary='symm'))
        pyramid[i + 1] = pyramid[i + 1][::2, ::2]
    return pyramid

def warp_image_eff(img, u,v, rot_flag = False):
    u_ave, v_ave = np.average(u[u != 0]), np.average(v[v != 0])
    transform = np.array([u_ave, v_ave])
    transform = np.hstack((np.eye(2, 2), transform.reshape(2, 1)))
    if rot_flag:
        da = np.arctan2(v_ave, u_ave)
        transform[0, 0] = 1-0.1* np.cos(da)
        transform[0, 1] = 0.1* -np.sin(da)
        transform[1, 0] = 0.1* np.sin(da)
        transform[1, 1] = 1 - 0.1* np.cos(da)
    I2_warped = cv2.warpAffine(img, transform, img.shape[0:2][::-1],cv2.INTER_LINEAR)
    I2_warped = np.reshape(I2_warped, img.shape)
    return I2_warped

def get_transform(img, u,v):
    u_ave, v_ave = np.average(u[u != 0]), np.average(v[v != 0])
    transform = np.array([u_ave, v_ave])
    transform = np.hstack((np.eye(2, 2), transform.reshape(2, 1)))

    return transform



def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Warp image using the optical flow parameters in u and v.
    Note that this method needs to support the case where u and v shapes do
    not share the same shape as of the image. We will update u and v to the
    shape of the image. The way to do it, is to:
    (1) cv2.resize to resize the u and v to the shape of the image.
    (2) Then, normalize the shift values according to a factor. This factor
    is the ratio between the image dimension and the shift matrix (u or v)
    dimension (the factor for u should take into account the number of columns
    in u and the factor for v showarp_imageuld take into account the number of rows in v).
    As for the warping, use `scipy.interpolate`'s `griddata` method. Define the
    grid-points using a flattened version of the `meshgrid` of 0:w-1 and 0:h-1.
    The values here are simply image.flattened().
    The points you wish to interpolate are, again, a flattened version of the
    `meshgrid` matrices - don't forget to add them v and u.
    Use `np.nan` as `griddata`'s fill_value.
    Finally, fill the nan holes with the source image values.
    Hint: For the final step, use np.isnan(image_warp).
    Args:
        image: np.ndarray. Image to warp.
        u: np.ndarray. Optical flow parameters corresponding to the columns.
        v: np.ndarray. Optical flow parameters corresponding to the rows.
    Returns:
        image_warp: np.ndarray. Warped image.
    """
    image_warp = image.copy()
    """INSERT YOUR CODE HERE.
    Replace image_warp with something else.
    """
    uv_list = [u.copy(),v.copy()]
    for i, mat in enumerate([u,v]):
        if image.shape != mat.shape:
            factor = image.shape[1] / mat.shape[1]
            uv_list[i] = cv2.resize(mat, image.T.shape) * factor
    u_new, v_new = uv_list[0], uv_list[1]
    y, x = image.shape
    y, x = np.arange(y), np.arange(x)
    xx, yy = np.meshgrid(x,y, indexing='xy')
    u_new += xx
    v_new += yy
    u_new, v_new = u_new.flatten(), v_new.flatten()
    start = time.time()
    interpolation_result = griddata((xx.flatten(), yy.flatten()), image.flatten(), (u_new.flatten(), v_new.flatten()), method='linear', fill_value=np.nan)
    end = time.time()
    # print(f'{end - start:.4f}[sec]')
    image_warp = interpolation_result.reshape(image.shape)
    image_warp[np.isnan(image_warp)] = image[np.isnan(image_warp)]
    return image_warp


def fixBorder(frame):
    h, w = frame.shape[0],frame.shape[1]
    # Scale the image 10% without moving the center
    T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.1)
    frame = cv2.warpAffine(frame, T, (w, h))#todo: check
    return frame

def lucas_kanade_step(I1: np.ndarray,
                      I2: np.ndarray,
                      window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform one Lucas-Kanade Step.
    This method receives two images as inputs and a window_size. It
    calculates the per-pixel shift in the x-axis and y-axis. That is,
    it outputs two maps of the shape of the input images. The first map
    encodes the per-pixel optical flow parameters in the x-axis and the
    second in the y-axis.
    (1) Calculate Ix and Iy by convolving I2 with the appropriate filters (
    see the constants in the head of this file).
    (2) Calculate It from I1 and I2.
    (3) Calculate du and dv for each pixel:
      (3.1) Start from all-zeros du and dv (each one) of size I1.shape.
      (3.2) Loop over all pixels in the image (you can ignore boundary pixels up
      to ~window_size/2 pixels in each side of the image [top, bottom,
      left and right]).
      (3.3) For every pixel, pretend the pixel’s neighbors have the same (u,
      v). This means that for NxN window, we have N^2 equations per pixel.
      (3.4) Solve for (u, v) using Least-Squares solution. When the solution
      does not converge, keep this pixel's (u, v) as zero.
    For detailed Equations reference look at slides 4 & 5 in:
    http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf
    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
    Returns:
        (du, dv): tuple of np.ndarray-s. Each one is of the shape of the
        original image. dv encodes the optical flow parameters in rows and du
        in columns.
    """
    """INSERT YOUR CODE HERE.
    Calculate du and dv correctly.
    """
    start_time = time.time()
    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    Ix = signal.convolve2d(I2,X_DERIVATIVE_FILTER,mode='same',boundary='symm')
    Iy = signal.convolve2d(I2,Y_DERIVATIVE_FILTER,mode='same',boundary='symm')
    It = I2.astype('int16') - I1.astype('int16') # need to cast to int, as the images are unsigned int8
    border_size = window_size // 2
    Ix_windowed = np.lib.stride_tricks.sliding_window_view(Ix,(window_size,window_size))
    Iy_windowed = np.lib.stride_tricks.sliding_window_view(Iy,(window_size,window_size))
    It_windowed = np.lib.stride_tricks.sliding_window_view(It,(window_size,window_size))
    for i in range(Ix_windowed.shape[0]):
        for j in range(Ix_windowed.shape[1]):

            A = np.vstack((Ix_windowed[i,j].ravel(), Iy_windowed[i,j].ravel())).T  # [Ix_cs, Iy_cs]
            b = -It_windowed[i,j].ravel()  # -It_windowed_cols : -It[p1..pk]^T

            At_A = np.matmul(A.transpose(), A)  # A^T * A
            At_b = np.matmul(A.transpose(), b)  # A^T * A
            try:
                du[i+border_size][j+border_size], dv[i+border_size][j+border_size] = np.linalg.lstsq(At_A, At_b, rcond=-1)[0]
            except np.linalg.LinAlgError:
                du[i+border_size][j+border_size], dv[i+border_size][j+border_size] = 0, 0
    end_time = time.time()
    return du, dv


def faster_lucas_kanade_step(I1: np.ndarray,
                             I2: np.ndarray,
                             window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Faster implementation of a single Lucas-Kanade Step.
    (1) If the image is small enough (you need to design what is good
    enough), simply return the result of the good old lucas_kanade_step
    function.
    (2) Otherwise, find corners in I2 and calculate u and v only for these
    pixels.
    (3) Return maps of u and v which are all zeros except for the corner
    pixels you found in (2).
    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
    Returns:
        (du, dv): tuple of np.ndarray-s. Each one of the shape of the
        original image. dv encodes the shift in rows and du in columns.
    """
    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    """INSERT YOUR CODE HERE.
    Calculate du and dv correctly.
    """
    Ix = signal.convolve2d(I2,X_DERIVATIVE_FILTER,mode='same',boundary='symm')
    Iy = signal.convolve2d(I2,Y_DERIVATIVE_FILTER,mode='same',boundary='symm')
    It = I2.astype('int16') - I1.astype('int16')
    if I1.shape[0]<=(window_size*10) or I1.shape[1]<=(window_size*10):
        return lucas_kanade_step(I1, I2, window_size)
    else:
        I2_temp=np.uint8(I2)
        corners = cv2.goodFeaturesToTrack(I2_temp, maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=window_size)
        try:
            corners = np.int0(corners)
        except:
            return du,dv

        for corner in corners:
            x, y = corner.ravel()
            Ix_windowed=Ix[y-window_size//2:y+window_size//2+1,x-window_size//2:x+window_size//2+1]
            Iy_windowed=Iy[y-window_size//2:y+window_size//2+1,x-window_size//2:x+window_size//2+1]
            It_windowed=It[y-window_size//2:y+window_size//2+1,x-window_size//2:x+window_size//2+1]
            A = np.vstack((Ix_windowed.ravel(), Iy_windowed.ravel())).T  # [Ix_cs, Iy_cs]
            b = -It_windowed.ravel()  # -It_windowed_cols : -It[p1..pk]^T
            At_A = np.matmul(A.transpose(), A)  # A^T * A
            At_b = np.matmul(A.transpose(), b)  # A^T * A
            try:
                du[y][x], dv[y][x] = np.linalg.lstsq(At_A, At_b, rcond=-1)[0]
            except np.linalg.LinAlgError:
                du[y][x], dv[y][x] = 0, 0
    return du, dv

def faster_lucas_kanade_optical_flow(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels .
    Use faster_lucas_kanade_step instead of lucas_kanade_step.
    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the shift in rows and u in columns.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** num_levels)))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** num_levels)))
    IMAGE_SIZE = (w_factor * (2 ** num_levels),
                  h_factor * (2 ** num_levels))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    # pyramid_I1 = build_pyramid(I1, num_levels)  # create levels list for I1
    # pyarmid_I2 = build_pyramid(I2, num_levels)  # create levels list for I1
    # u = np.zeros(pyarmid_I2[-1].shape)  # create u in the size of smallest image
    # v = np.zeros(pyarmid_I2[-1].shape)  # create v in the size of smallest image
    """INSERT YOUR CODE HERE.
    Replace u and v with their true value."""
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                  h_factor * (2 ** (num_levels - 1 + 1)))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyarmid_I2 = build_pyramid(I2, num_levels)
    # start from u and v in the size of smallest image
    u = np.zeros(pyarmid_I2[-1].shape)
    v = np.zeros(pyarmid_I2[-1].shape)
    """INSERT YOUR CODE HERE.
       Replace u and v with their true value."""
    for level in range(num_levels, -1, -1):
        if level >= 2:
            I2_warped = warp_image(pyarmid_I2[level], u, v)
        else:
            I2_warped = warp_image_eff(pyarmid_I2[level], u, v)

        # I2_warped = warp_image_eff(pyarmid_I2[level], u, v)
        I2_warped = np.reshape(I2_warped, pyramid_I1[level].shape)
        for k in range(max_iter):
            du, dv = faster_lucas_kanade_step(pyramid_I1[level], I2_warped, window_size)
            u += du
            v += dv

            start = time.time()
            if level <= 2:
                u_ave, v_ave = np.average(u[u != 0]), np.average(v[v != 0])
                transform = np.array([u_ave, v_ave])
                transform = np.hstack((np.eye(2, 2), transform.reshape(2, 1)))
                # da = np.arctan2(v_ave, u_ave)
                # transform[0, 0] = np.cos(da)
                # transform[0, 1] = -np.sin(da)
                # transform[1, 0] = np.sin(da)
                # transform[1, 1] = np.cos(da)
                I2_warped = cv2.warpAffine(pyramid_I1[level], transform, pyramid_I1[level].shape[::-1])
                I2_warped = np.reshape(I2_warped, pyramid_I1[level].shape)
            else:
                I2_warped = warp_image(pyarmid_I2[level], u, v)
            end = time.time()
            # print(f'{end - start:.4f}[sec]')
        if level > 0:
            dim = (2 * u.shape[1], 2 * u.shape[0])
            u = 2 * cv2.resize(u, dim)
            v = 2 * cv2.resize(v, dim)
    return u, v

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

def lucas_kanade_faster_video_stabilization(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.
    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
    Returns:
        None.
    """
    """INSERT YOUR CODE HERE."""
    cap, out = get_video_files(input_video_path, output_video_path, isColor=False)
    ret, prevframe = cap.read()
    cap_color, out_color = get_video_files(input_video_path, output_video_path, isColor=True)
    ret_color, prevframe_color = cap_color.read()

    prevframe = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)
    prevframe = cv2.resize(prevframe, (270,331), interpolation=cv2.INTER_CUBIC)
    K = int(np.ceil(prevframe.shape[0] / (2 ** (num_levels - 1))))
    M = int(np.ceil(prevframe.shape[1] / (2 ** (num_levels - 1))))
    M *= int(2 ** (num_levels - 1))
    K *= int(2 ** (num_levels - 1))
    IMAGE_SIZE = (K, M)
    prevframe = cv2.resize(prevframe, IMAGE_SIZE)
    prev_u, prev_v = np.zeros(IMAGE_SIZE), np.zeros(IMAGE_SIZE)
    prev_u = cv2.resize(prev_u, IMAGE_SIZE)
    prev_v = cv2.resize(prev_v, IMAGE_SIZE)
    array_of_frame=[]

    u_av, v_av = -1, -1
    last_u_av, last_v_av = 0, 0

    frames_bgr = load_entire_video(cap_color, color_space='bgr')
    frames_bw = load_entire_video(cap, color_space='bw')

    # Pre-define transformation-store array
    transforms = np.zeros((len(frames_bgr) , 2), np.float32)

    # for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc=f"frame "):
    for i, next_frame_color in tqdm(enumerate(frames_bgr), desc=f"frame "):
        if i == 0:
            continue
        next_frame = frames_bw[i]
        # ret, next_frame = cap.read()
        # ret_color, next_frame_color = cap_color.read()

        # if ret:
        # if i % 5 == 0 or np.abs(u_av - last_u_av) >= 0.75 or np.abs(v_av - last_v_av) >= 0.75:
        last_u_av, last_v_av = u_av, v_av
        # next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("result_temp/frame_old%d.jpg" % (i), next_frame)#todo:delete

        if next_frame.shape != IMAGE_SIZE:
            next_frame = cv2.resize(next_frame, IMAGE_SIZE)
        (u, v) = faster_lucas_kanade_optical_flow(prevframe, next_frame, window_size, max_iter, num_levels)
        scale_u, scale_v = prevframe_color.shape[0]/270.0 , prevframe_color.shape[0]/331.0
        u_av, v_av = scale_u*np.average(u[u!=0]), scale_v*np.average(v[v!=0])
        u, v = np.ones(u.shape) * u_av, np.ones(v.shape) * v_av
        if u.shape != IMAGE_SIZE:
            u = cv2.resize(u, IMAGE_SIZE)
        if v.shape != IMAGE_SIZE:
            v = cv2.resize(v, IMAGE_SIZE)
        # output_frame = warp_image(next_frame, u + prev_u, v + prev_v)

        # output_frame = warp_image_eff(next_frame, u + prev_u, v + prev_v)
        # output_frame = warp_image_eff(next_frame_color, (u + prev_u), (v + prev_v), rot_flag=True)

        trans = get_transform(next_frame_color, (u + prev_u), (v + prev_v))
        # Extract traslation
        dx = trans[0, 2]
        dy = trans[1, 2]

        # Extract rotation angle
        # da = np.arctan2(trans[1, 0], trans[0, 0])

        transforms[i] = [dx,dy]

        # output_frame = fixBorder(output_frame) #TODO: test

        # output_frame = cv2.resize(output_frame, (prevframe.shape[1], prevframe.shape[0]))
        # array_of_frame.append(output_frame)
        # print frame as a jpg#todo:delete
        # print frame as a jpg#todo:delete

        # cv2.imwrite("result_temp/frame_new%d.jpg" % (i), output_frame)#todo:delete
        prev_u, prev_v = u + prev_u, v + prev_v
        prevframe = next_frame
        # else:
        #     next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        #     cv2.imwrite("result_temp/frame_old%d.jpg" % (i), next_frame)#todo:delete
        #
        #     if next_frame.shape != IMAGE_SIZE:
        #         next_frame = cv2.resize(next_frame, IMAGE_SIZE)
        #     output_frame = warp_image_eff(next_frame_color, u + prev_u, v + prev_v)
        #     array_of_frame.append(output_frame)
        #
        #     cv2.imwrite("result_temp/frame_new%d.jpg" % (i), output_frame)#todo:delete
        #     prevframe = next_frame

        # else:
        #     break
        # if(i==2):#todo:delete
        #     break#todo:delete
        # i += 1


    # The larger the more stable the video, but less reactive to sudden panning
    SMOOTHING_RADIUS = 50
    def movingAverage(curve, radius):
        window_size = 2 * radius + 1
        # Define the filter
        f = np.ones(window_size) / window_size
        # Add padding to the boundaries
        curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
        # Apply convolution
        curve_smoothed = np.convolve(curve_pad, f, mode='same')
        # Remove padding
        curve_smoothed = curve_smoothed[radius:-radius]
        # return smoothed curve
        return curve_smoothed

    def smooth(trajectory):
        smoothed_trajectory = np.copy(trajectory)
        # Filter the x, y and angle curves
        for i in range(2):
            smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)

        return smoothed_trajectory
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    # Create variable to store smoothed trajectory
    smoothed_trajectory = smooth(trajectory)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    for i, transform in enumerate(transforms):
        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        # da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = 1
        m[0, 1] = 0
        m[1, 0] = 0
        m[1, 1] = 1
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frames_bgr[i], m, frames_bgr[i].shape[0:2][::-1], cv2.INTER_LINEAR)

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)

        array_of_frame.append(frame_stabilized)

    array_of_frame_to_avi_file(array_of_frame, output_video_path,cap_color.get(cv2.CAP_PROP_FPS))
    cap.release()
    out.release()
    cap_color.release()
    out_color.release()
    cv2.destroyAllWindows()
    return None







# def get_video_files(path):
#     cap = cv2.VideoCapture(path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     return cap, width, height, fps
#
#





# def movingAverage(curve, radius):
#     window_size = 2 * radius + 1
#     # Define the filter
#     f = np.ones(window_size) / window_size
#     # Add padding to the boundaries
#     curve_pad = np.lib.pad(curve, (radius, radius), 'reflect')
#
#     '''Fix padding manually'''
#     for i in range(radius):
#         curve_pad[i] = curve_pad[radius] - curve_pad[i]
#
#     for i in range(len(curve_pad) - 1, len(curve_pad) - 1 - radius, -1):
#         curve_pad[i] = curve_pad[len(curve_pad) - radius - 1] - curve_pad[i]
#
#     # Apply convolution
#     curve_smoothed = np.convolve(curve_pad, f, mode='same')
#     # Remove padding
#     curve_smoothed = curve_smoothed[radius:-radius]
#     return curve_smoothed
#
#
# def smooth(trajectory, smooth_radius):
#     smoothed_trajectory = np.copy(trajectory)
#     for i in range(smoothed_trajectory.shape[1]):
#         smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=smooth_radius)
#     return smoothed_trajectory
#
#
# def write_video(output_path, frames, fps, out_size, is_color):
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     video_out = cv2.VideoWriter(output_path, fourcc, fps, out_size, isColor=is_color)
#     for frame in frames:
#         video_out.write(frame)
#     video_out.release()
#
# def load_entire_video(cap, color_space='bgr'):
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frames = []
#     for i in range(n_frames):
#         success, curr = cap.read()
#         if not success:
#             break
#         if color_space == 'bgr':
#             frames.append(curr)
#         elif color_space == 'yuv':
#             frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2YUV))
#         elif color_space == 'bw':
#             frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY))
#         else:
#             frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2HSV))
#         continue
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     return np.asarray(frames)
#
# def stabilize_video(input_video_path, output_video_path="../Temp/stabilized_{0}_{0}.avi".format(ID1,ID2)):
#     my_logger.info('Starting Video Stabilization')
#     cap, w, h, fps = get_video_files(path=input_video_path)
#     frames_bgr = load_entire_video(cap, color_space='bgr')
#     n_frames = len(frames_bgr)
#     prev = frames_bgr[0]
#     prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
#
#     transforms = np.zeros((n_frames - 1, 9), np.float32)
#     transforms_list = np.zeros((n_frames - 1, 3, 3), np.float32)
#
#
#     for  frame_index in tqdm(range(1,n_frames-1), desc='Stabilizing Video Frames number'):
#
#         prev_pts = cv2.goodFeaturesToTrack(prev_gray,
#                                            maxCorners=MAX_CORNERS,
#                                            qualityLevel=QUALITY_LEVEL,
#                                            minDistance=MIN_DISTANCE,
#                                            blockSize=BLOCK_SIZE)
#         curr_gray = cv2.cvtColor(frames_bgr[frame_index], cv2.COLOR_BGR2GRAY)
#         curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)#todo: check
#
#         # Filter only valid points
#         idx = np.where(status == 1)[0]
#         prev_pts, curr_pts = prev_pts[idx], curr_pts[idx]
#
#         # Find transformation matrix
#         transform_matrix, _ = cv2.findHomography(prev_pts, curr_pts)#todo: check
#         transforms[frame_index] = transform_matrix.flatten()
#         prev_gray = curr_gray
#         frame_index+=1
#
#     # Compute trajectory using cumulative sum of transformations
#     trajectory = np.cumsum(transforms, axis=0)
#
#     smoothed_trajectory = smooth(trajectory, SMOOTH_RADIUS)
#     # Calculate difference in smoothed_trajectory and trajectory
#     difference = smoothed_trajectory - trajectory
#
#     # Calculate smooth transformation array
#     transforms_smooth = transforms + difference
#
#     stabilized_frames_list = [frames_bgr[0]]
#     stabilized_warped_frames_list = [frames_bgr[0]]
#     # Write n_frames-1 transformed frames
#     for frame_index, frame in tqdm(enumerate(frames_bgr[:-1]), desc='Applying warps Video Frames number'):
#         #print(f'[Video Stabilization] Applying warps to frame: {frame_index+1} / {n_frames - 1}')
#         # Apply affine wrapping to the given frame
#         #save frame to jpg file
#         cv2.imwrite(f'../Temp/frame_{frame_index}.jpg', frame)
#         transform_matrix = transforms_smooth[frame_index].reshape((3, 3))
#         frame_stabilized = cv2.warpPerspective(frames_bgr[frame_index], transform_matrix, (w, h))
#         stabilized_warped_frames_list.append(frame_stabilized)
#         frame_stabilized = fixBorder(frame_stabilized)
#         stabilized_frames_list.append(frame_stabilized)
#         cv2.imwrite(f'../Temp/frame_stable{frame_index}.jpg', frame_stabilized)
#         cv2.imwrite(f'../Temp/frame_diff{frame_index}.jpg', frame_stabilized-frame)
#         transforms_list[frame_index] = transform_matrix
#
#     cap.release()
#     cv2.destroyAllWindows()
#     write_video(output_video_path, stabilized_frames_list, fps, (w, h), is_color=True)
#     output_warp = "../Temp/warped_{0}_{0}.avi".format(ID1,ID2)
#     write_video(output_warp, stabilized_warped_frames_list, fps, (w, h), is_color=True)
#     transforms_list.dump('../Temp/transforms_video_stab.np')
#     print('~~~~~~~~~~~ [Video Stabilization] FINISHED! ~~~~~~~~~~~')
#     print('~~~~~~~~~~~ stabilize.avi has been created! ~~~~~~~~~~~')
#     my_logger.info('Finished Video Stabilization')
#
# def main():
#     stabilize_video('../Temp/INPUT.avi')



def main():
    # Load video file
    input_video_name = '../Temp/INPUT.avi'
    faster_output_video_name = f'../Temp/{ID1}_{ID2}_faster_stabilized_video.avi'
    start_time = time.time()
    lucas_kanade_faster_video_stabilization(input_video_name, faster_output_video_name, WINDOW_SIZE_TAU, MAX_ITER_TAU,
                                            NUM_LEVELS_TAU)
    end_time = time.time()
    print(f'LK-Video Stabilization FASTER implementation took: '
          f'{end_time - start_time:.2f}[sec]')

if __name__ == '__main__':
    main()
