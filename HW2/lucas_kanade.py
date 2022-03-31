import cv2
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import griddata


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
    pyramid += [signal.convolve2d(PYRAMID_FILTER, pyramid[i], mode='same', boundary='symm')[::2] for i in range(num_levels)]
    return pyramid


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
    def _column_stack(matrix):
        return np.transpose(np.transpose(matrix).reshape([1, len(matrix) * len(matrix[0])]))

    def build_windowed_B(Ix, Iy, i, j, window_size=5):
        half_window = window_size // 2
        temp_Ix_cs = _column_stack(Ix[max(0, i - half_window):min(i + half_window + 1, len(Ix)),
                                   max(0, j - half_window):min(j + half_window + 1, len(Ix[0]))])
        temp_Iy_cs = _column_stack(Iy[max(0, i - half_window):min(i + half_window + 1, len(Iy)),
                                   max(0, j - half_window):min(j + half_window + 1, len(Iy[0]))])
        B = np.column_stack((temp_Ix_cs, temp_Iy_cs))
        return B

    def build_windowed_it(It, i, j, window_size):
        half_window = window_size // 2
        return _column_stack(It[max(0, i - half_window):min(i + half_window + 1, len(It)),
                             max(0, j - half_window):min(j + half_window + 1, len(It[0]))])
    def get_derivatives(I2):
        deriv_X_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        deriv_Y_filter = deriv_X_filter.copy().transpose()
        Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, mode='same',boundary='symm')
        Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, mode='same',boundary='symm')
        return Ix, Iy

    def calc_pixel_delta_p(It, Ix, Iy, i, j, window_size=5):
        B_windowed = build_windowed_B(Ix, Iy, i, j, window_size)
        It_cs = build_windowed_it(It, i, j, window_size)
        try:
            B_traspose_B = np.matmul(np.transpose(B_windowed), B_windowed)
            delta_p = -np.matmul(np.matmul(np.linalg.inv(B_traspose_B), np.transpose(B_windowed)), It_cs)
        except np.linalg.LinAlgError:
            return np.zeros((2, 1))
        return delta_p

    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    Ix = signal.convolve2d(I2,X_DERIVATIVE_FILTER,mode='same',boundary='symm')
    Iy = signal.convolve2d(I2,Y_DERIVATIVE_FILTER,mode='same',boundary='symm')
    It = I2.astype('int16') - I1.astype('int16') # need to cast to int, as the images are unsigned int8
    border_size = window_size // 2
    for i in range(border_size, len(I2) - border_size):
        for j in range(border_size, len(I2[0]) - border_size):
            A_1 = Ix[i-border_size:i+border_size+1, j-border_size: j+border_size+1].flatten().transpose()  # Ix_windowed_cols : Ix[p1..pk]^T
            A_2 = Iy[i-border_size:i+border_size+1, j-border_size: j+border_size+1].flatten().transpose()  # Iy_windowed_cols : Iy[p1..pk]^T
            A = np.column_stack((A_1, A_2))  # [Ix_cs, Iy_cs]
            b = -It[i-border_size:i+border_size+1, j-border_size: j+border_size+1].flatten().transpose() # -It_windowed_cols : -It[p1..pk]^T
            At_A = np.matmul(A.transpose(), A)  # A^T * A
            At_b = np.matmul(A.transpose(), b)  # A^T * A
            try:
                # delta_p = calc_pixel_delta_p(It, Ix, Iy, i, j, window_size)
                du[i][j],dv[i][j] = np.linalg.lstsq(At_A, At_b, rcond=-1)[0]
            except np.linalg.LinAlgError:
                du[i][j], dv[i][j] = 0,0
            #
            # delta_p = calc_pixel_delta_p(It, Ix, Iy, i, j, window_size)
            # if np.max(np.abs(delta_p)) == 0:
            #     break
            # du[i][j] += delta_p[0][0]
            # dv[i][j] += delta_p[1][0]



    # du = np.zeros(I2.shape)
    # dv = np.zeros(I2.shape)
    # Ix, Iy = get_derivatives(I2)
    #
    # for i in range(border_size, len(I2) - border_size):
    #     for j in range(border_size, len(I2[0]) - border_size):
    #         delta_p = calc_pixel_delta_p(It, Ix, Iy, i, j, window_size)
    #         if np.max(np.abs(delta_p)) == 0:
    #             break
    #         du[i][j] += delta_p[0][0]
    #         dv[i][j] += delta_p[1][0]
    return du, dv


def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Warp image using the optical flow parameters in u and v.

    Note that this method needs to support the case where u and v shapes do
    not share the same shape as of the image. We will update u and v to the
    shape of the image. The way to do it, is to:
    (1) cv2.resize to resize the u and v to the shape of the image.
    (2) Then, normalize the shift values according to a factor. This factor
    is the ratio between the image dimension and the shift matrix (u or v)
    dimension (the factor for u should take into account the number of columns
    in u and the factor for v should take into account the number of rows in v).

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
    uv_list = [u,v]
    for i, mat in enumerate([u,v]):
        if image.shape != mat.shape:
            factor = image.shape[1] / mat.shape[1]
            uv_list[i] = cv2.resize(mat, image.T.shape) * factor
    u_new, v_new = uv_list[0], uv_list[1]
    # x_1 = []
    # y_1 = []
    # u_1 = []
    # v_1 = []
    # values = []
    # new_image = np.zeros(image.shape)
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         x_1.append(j)
    #         y_1.append(i)
    #         values.append(image[i][j])
    #         u_1.append(j + u[i][j])
    #         v_1.append(i + v[i][j])
    # x,y = image.shape
    # x, y = np.arange(x), np.arange(y)
    # xx, yy = np.meshgrid(x,y)
    # image_flat = image.flatten()
    # u_new += xx.transpose()
    # v_new += yy.transpose()
    # u_new, v_new = u_new.flatten(), v_new.flatten()
    # bilinear_result = griddata((xx.flatten(), yy.flatten()), image_flat, (u_new.flatten(), v_new.flatten()), method='linear', fill_value=np.nan)
    # image_warp = bilinear_result.reshape(image.shape)
    # image_warp[np.isnan(image_warp)] = image[np.isnan(image_warp)]
    x_1 = []
    y_1 = []
    u_1 = []
    v_1 = []
    values = []
    new_image = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x_1.append(j)
            y_1.append(i)
            values.append(image[i][j])
            u_1.append(j + u[i][j])
            v_1.append(i + v[i][j])
    bilinear_result = griddata((x_1, y_1), values, (u_1, v_1), method='linear')
    new_image = bilinear_result.reshape(image.shape)
    # for index, value in enumerate(bilinear_result):
    #     i = index // image.shape[1]
    #     j = index % image.shape[1]
    #     new_image[i][j] = value

    # WHEN PIXELS ARE GONE, THEY RECEIVE NAN, so we put there their original value
    gone_pixels = np.argwhere(np.isnan(new_image))
    if len(gone_pixels) > 0:
        for pixel_coords in gone_pixels:
            new_image[pixel_coords[0]][pixel_coords[1]] = image[pixel_coords[0]][pixel_coords[1]]
    image_warp = new_image
    return image_warp


def lucas_kanade_optical_flow(I1: np.ndarray,
                              I2: np.ndarray,
                              window_size: int,
                              max_iter: int,
                              num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the optical flow parameters in rows and u in
        columns.

    Recipe:
        (1) Since the image is going through a series of decimations,
        we would like to resize the image shape to:
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (2) Build pyramids for the two images.
        (3) Initialize u and v as all-zero matrices in the shape of I1.
        (4) For every level in the image pyramid (start from the smallest
        image):
          (4.1) Warp I2 from that level according to the current u and v.
          (4.2) Repeat for num_iterations:
            (4.2.1) Perform a Lucas Kanade Step with the I1 decimated image
            of the current pyramid level and the current I2_warp to get the
            new I2_warp.
          (4.3) For every level which is not the image's level, perform an
          image resize (using cv2.resize) to the next pyramid level resolution
          and scale u and v accordingly.
    """
    """INSERT YOUR CODE HERE.
        Replace image_warp with something else.
        """
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
    u = np.zeros(I1.shape)
    v = np.zeros(I1.shape)
    return u, v


def lucas_kanade_video_stabilization(input_video_path: str,
                                     output_video_path: str,
                                     window_size: int,
                                     max_iter: int,
                                     num_levels: int) -> None:
    """Use LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.

    Recipe:
        (1) Open a VideoCapture object of the input video and read its
        parameters.
        (2) Create an output video VideoCapture object with the same
        parameters as in (1) in the path given here as input.
        (3) Convert the first frame to grayscale and write it as-is to the
        output video.
        (4) Resize the first frame as in the Full-Lucas-Kanade function to
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (5) Create a u and a v which are og the size of the image.
        (6) Loop over the frames in the input video (use tqdm to monitor your
        progress) and:
          (6.1) Resize them to the shape in (4).
          (6.2) Feed them to the lucas_kanade_optical_flow with the previous
          frame.
          (6.3) Use the u and v maps obtained from (6.2) and compute their
          mean values over the region that the computation is valid (exclude
          half window borders from every side of the image).
          (6.4) Update u and v to their mean values inside the valid
          computation region.
          (6.5) Add the u and v shift from the previous frame diff such that
          frame in the t is normalized all the way back to the first frame.
          (6.6) Save the updated u and v for the next frame (so you can
          perform step 6.5 for the next frame.
          (6.7) Finally, warp the current frame with the u and v you have at
          hand.
          (6.8) We highly recommend you to save each frame to a directory for
          your own debug purposes. Erase that code when submitting the exercise.
       (7) Do not forget to gracefully close all VideoCapture and to destroy
       all windows.
    """
    """INSERT YOUR CODE HERE."""
    pass


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
    pyramid_I1 = build_pyramid(I1, num_levels)  # create levels list for I1
    pyarmid_I2 = build_pyramid(I2, num_levels)  # create levels list for I1
    u = np.zeros(pyarmid_I2[-1].shape)  # create u in the size of smallest image
    v = np.zeros(pyarmid_I2[-1].shape)  # create v in the size of smallest image
    """INSERT YOUR CODE HERE.
    Replace u and v with their true value."""
    u = np.zeros(I1.shape)
    v = np.zeros(I1.shape)
    return u, v


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
    pass


def lucas_kanade_faster_video_stabilization_fix_effects(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int, start_rows: int = 10,
        start_cols: int = 2, end_rows: int = 30, end_cols: int = 30) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
        start_rows: int. The number of lines to cut from top.
        end_rows: int. The number of lines to cut from bottom.
        start_cols: int. The number of columns to cut from left.
        end_cols: int. The number of columns to cut from right.

    Returns:
        None.
    """
    """INSERT YOUR CODE HERE."""
    pass


