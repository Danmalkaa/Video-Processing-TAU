import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches



# change IDs to your IDs.
ID1 = "313325938"
ID2 = "304773591"

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,  # x center
             139,  # y center
             16,  # half width
             43,  # half height
             0,  # velocity x
             0]  # velocity y
MU = 0
X = 2
Y = 2
X_V = 0.8
Y_V = 0.8

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
    state_drifted[:2, :] = state_drifted[:2, :] + state_drifted[4:, :]

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
    fig, ax = plt.subplots(1)
    image = image[:,:,::-1]
    plt.imshow(image)
    plt.title(ID + " - Frame mumber = " + str(frame_index))

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
    y_avg = y_avg - state[3][0]

    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    (x_max, y_max, w_max, h_max) = (0, 0, 0, 0)
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""


    x_max,y_max=state.T[np.argmax(W)][0], state.T[np.argmax(W)][1]

    w_max=state[2][0] * 2
    h_max=state[3][0] * 2
    x_max = x_max - state[2][0]
    y_max = y_max - state[3][0]
    rect = patches.Rectangle((x_max, y_max), w_max, h_max, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    #plt.show(block=False)#todo:need to not del

    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
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




def main():

    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame)

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    # YOU NEED TO FILL THIS PART WITH CODE:
    """INSERT YOUR CODE HERE."""
    C, _ = calc_C_weights(image, S, q)
    images_processed = 1
    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    for image_name in sorted(image_name_list)[1:]:
        S_prev = S
        # LOAD NEW IMAGE FRAME
        image_path = IMAGE_DIR_PATH + os.sep + image_name
        current_image = cv2.imread(image_path)
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
        if images_processed % 10 == 0:
            print("Processed image number: " + str(images_processed))
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)
    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)
if __name__ == "__main__":
    main()
