import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm



# change IDs to your IDs.
ID1 = "313325931"
ID2 = "304773591"

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
os.makedirs("../Temp/tracking_frames", exist_ok=True)
IMAGE_DIR_PATH = "Images"
INPUT_VIDEO_PATH = "../Temp/stabilized_304773591_304773591.avi"
SAVE_FRAME_INPUT="../Temp/tracking_frames"
SAVE_FRAME_OUTPUT="../Temp/tracking_frames/"
OUTPUT_VIDEO="../Outputs/tracking_"+ID1+"_"+ID2+".avi"
# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [180,  # x center
             300,  # y center
             20,  # half width
             43,  # half height
             0,  # velocity x
             0]  # velocity y
MU = 0
X = 3
Y = 0.5
X_V = 1.2
Y_V = 0.5

def creatFrameArrayFromVideo(video_path,save=False,save_path=None):
    cap = cv2.VideoCapture(video_path)
    frame_array = []
    i=0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_array.append(frame)
            if save:
                cv2.imwrite(save_path + "/" + "frame_" + str(i).zfill(3) + ".png", frame_array[i])
        else:
            break
        i+=1
    cap.release()
    #save the array of frames as a png image

    return frame_array
def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    s_prior = s_prior.astype(float)
    state_drifted = s_prior
    """ DELETE THE LINE ABOVE AND:
    INSERT YOUR CODE HERE."""
    state_drifted[:2, :] = state_drifted[:2, :] + state_drifted[4:, :] # add x,y velocities


    state_drifted[:1, :] = state_drifted[:1, :] + \
                              np.round(np.random.normal(MU, X, size=(1, 100)))
    state_drifted[1:2, :] = state_drifted[1:2, :] + \
                               np.round(np.random.normal(MU, Y, size=(1, 100)))
    state_drifted[4:5, :] = state_drifted[4:5, :] + \
                               np.round(np.random.normal(MU, X_V, size=(1, 100)))
    state_drifted[5:6, :] = state_drifted[5:6, :] + \
                               np.round(np.random.normal(MU, Y_V, size=(1, 100)))
    state_drifted = state_drifted.astype(int)
    return state_drifted
def covertFrameArrayToVideo(frameArray,outputVideoPath):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputVideoPath, fourcc, 30.0, (frameArray[0].shape[1], frameArray[0].shape[0]))
    for frame in frameArray:
        out.write(frame)
    out.release()
    return outputVideoPath
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
def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    state = np.floor(state)
    state = state.astype(int)
    hist = np.zeros((1, 16 * 16 * 16))
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    x, y, width, height, x_vel, y_vel = state

    temp_image = image[y - height:y + height+1, x - width:x + width+1,:]

    b, g, r = cv2.split(temp_image)

    b //= 16
    g //= 16
    r //= 16

    hist = np.zeros((16, 16, 16))

    for i in range(len(temp_image)):
        for j in range(len(temp_image[0])):
            hist[b[i, j]][g[i, j]][r[i, j]] += 1

    hist = hist.reshape((4096, 1))


    # normalize
    hist = hist/np.sum(hist)

    return hist


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """
    S_next = np.zeros(previous_state.shape)
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    sampled_particles = list()
    for i in range(len(previous_state[1])):
        r = np.random.uniform(0, 1)
        j = np.argmax(cdf >= r)
        sampled_particles.append(previous_state[:, j])

    S_next= np.array(sampled_particles).T
    return S_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """
    distance = 0
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    distance=np.exp(20 * np.sum(np.sqrt(p * q)))
    return distance


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:


    # Avg particle box
    (x_avg, y_avg, w_avg, h_avg) = (0, 0, 0, 0)
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    for index, particle in enumerate(state.T):
        x_avg += particle[0] * W[index]
        y_avg += particle[1] * W[index]
    w_avg= state[2,0]*2
    h_avg = state[3,0]*2
    x_avg=x_avg - state[2][0]
    y_avg = (y_avg) - state[3][0]

    x,y=(x_avg-0.9*(3*w_avg)),( y_avg-h_avg*1.1)
    x=int(x)
    y=int(y)
    w,h=(7*w_avg), (9.6*h_avg)
    w=int(w)
    h=int(h)
    y=y+h
    image2 = cv2.rectangle(image.astype("uint8"), (x, y), (x+w, y-h), (100, 0, 0), 2)

    # calculate Max particle box
    (x_max, y_max, w_max, h_max) = (0, 0, 0, 0)
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    x_max,y_max=state.T[np.argmax(W)][0], state.T[np.argmax(W)][1]
    w_max=state[2][0] * 2
    h_max=state[3][0] * 2
    x_max = x_max - state[2][0]
    y_max = y_max - state[3][0]
    x,y=(x_max-0.9*(3*w_max)),( y_max-h_max*1.1)
    x=int(x)
    y=int(y)
    w,h=(7*w_max), (9.6*h_max)
    w=int(w)
    h=int(h)
    y=y+h
    image2 = cv2.rectangle(image2.astype("uint8"), (x, y), (x+w, y-h), (0, 255, 0), 2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    cv2.imwrite(SAVE_FRAME_OUTPUT+ "frame_" + str(frame_index).zfill(3) + ".jpg", image2)
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state





def calc_C_weights(image, particles_list, real_histogram):
    """

    :param image:
    :param particles_list:
    :param real_histogram:
    :return:  cdf, weights of frame
    """
    weights = [bhattacharyya_distance(compute_normalized_histogram(image, column), real_histogram) for column in particles_list.T]

    weights = np.array(weights)
    weights /= np.sum(weights)

    C= np.zeros(len(weights))
    C[0] = weights[0]
    for i in range(1, len(weights)):
        C[i] = weights[i] + C[i - 1]

    return C, weights
def load_entire_video(cap, color_space='bgr'):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n_frames):
        success, curr = cap.read()
        if not success:
            break
        if color_space == 'bgr':
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2RGB))
        elif color_space == 'yuv':
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2YUV))
        elif color_space == 'bw':
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY))
        else:
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2HSV))
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.asarray(frames)

def main():
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    frame_array=load_entire_video(cap)
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame)

    # LOAD FIRST IMAGE
    # image_name_list = os.listdir(SAVE_FRAME_INPUT)
    # image_name_list.sort()
    # image = cv2.imread(SAVE_FRAME_INPUT+"/"+image_name_list[0])

    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(frame_array[2], s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    # YOU NEED TO FILL THIS PART WITH CODE:
    """INSERT YOUR CODE HERE."""
    C, _ = calc_C_weights(frame_array[2], S, q)
    images_processed = 1
    # MAIN TRACKING LOOP
    # image_name_list = os.listdir(IMAGE_DIR_PATH)
    # image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}

    for current_image in tqdm(frame_array[3:]):
        if len(np.nonzero(current_image)[0]) == 0:
            continue
        S_prev = S
        # LOAD NEW IMAGE FRAME

        # image_path = SAVE_FRAME_INPUT+"/" + image_name
        #current_image = cv2.imread(image_path)
        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)
        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predict_particles(S_next_tag)
        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        # YOU NEED TO FILL THIS PART WITH CODE:
        """INSERT YOUR CODE HERE."""
        C, W = calc_C_weights(current_image, S, q)
        # CREATE DETECTOR PLOTS
        images_processed += 1

        if images_processed % 1 == 0:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)


    with open(os.path.join("../Outputs", 'tracking_avg.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join("../Outputs", 'tracking_max.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)
    covertFrameArrayToVideo(reaf_all_frame_from_folser(SAVE_FRAME_OUTPUT), OUTPUT_VIDEO)
def tracking():
    main()

if __name__ == "__main__":

    main()

