import random
import time
from collections import deque

import cv2
import numpy as np
from PIL import ImageGrab
from keras.layers import Conv2D, Activation, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam

# model hyper parameters
LEARNING_RATE = 1e-4
img_rows, img_cols = 40, 20
img_channels = 4
ACTIONS = 2
# game parameters
GAMMA = 0.99  # decay rate of past observations original 0.99
OBSERVATION = 50000.  # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1


def grab_screen(_driver=None):
    # bbox = region of interest on the entire screen
    screen = np.array(ImageGrab.grab(bbox=(40, 180, 440, 400)))
    image = process_img(screen)
    return image


def process_img(image):
    # game is already in grey scale, canny to get only edges reduce unwanted objects (clouds)
    # rescale image dimensions
    image = cv2.resize(image, (0, 0), fx=0.15, fy=0.10)
    # crop out the dino agent from the frame
    image = image[2:38, 10:50]
    image = cv2.Canny(image, threshold1=100, threshold2=200)
    return image


def build_model():
    print('Now we build the model')
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same', input_shape=(img_cols, img_rows, img_channels)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    print('We finish building the model')
    return model


def train_network(model, game_state):
    """
    :param model: Keras Model to be trained
    :param game_state: Game State module with access to game environment and dino
    :return: flag to indicate whether the model is to be trained (weight updates), else just play
    """

    def train_batch(mini_batch):
        inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32, 20, 40, 4
        targets = np.zeros((inputs.shape[0], ACTIONS))  # 32, 2
        loss = 0

        for i in range(len(mini_batch)):
            state_t = mini_batch[i][0]  # 4D stack of images
            action_t = mini_batch[i][1]  # This is action index
            reward_t = mini_batch[i][2]  # reward at state_t due to action_t
            state_t1 = mini_batch[i][3]  # next state
            terminal = mini_batch[i][4]  # whether the agent died or survived due the action
            inputs[i:i + 1] = state_t
            targets[i] = model.predict(state_t)  # predicted q values
            Q_sa = model.predict(state_t1)  # predict q values for next step
            if terminal:
                targets[i, action_t] = reward_t  # if terminated, only equals reward
            else:
                targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss += model.train_on_batch(inputs, targets)

    # store the previous observations in replay memory
    D = deque()
    # get the first state by doing nothing
    do_nothing = np.zeros(ACTIONS)
    # 0 => do nothing | 1 => jump
    do_nothing[0] = 1

    # get next step after performing the action
    x_t, r_0, terminal = game_state.get_state(do_nothing)
    # stack 4 images to create placeholder input reshaped 1*20*40*4
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2).reshape(1, 20, 40, 4)

    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON
    t = 0
    while True:  # endless running
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0  # reward at t
        a_t = np.zeros([ACTIONS])  # action at t

        # choose an action epsilon greedy
        if random.random() <= epsilon:  # randomly explore an action
            print('---------Random Action---------')
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            q = model.predict(s_t)  # input a stack of 4 images, get the prediction
            max_Q = np.argmax(q)  # choosing index with maximum q value
            action_index = max_Q
            a_t[action_index] = 1  # 0 => do nothing | 1 => jump

        # We reduced the epsilon (exploration parameter) gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observed next state and reward
        x_t1, r_t, terminal = game_state.get_state(a_t)
        last_time = time.time()
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)  # append the new image to input stack and remove the first one

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing; sample a mini batch to train on
        if t > OBSERVE:
            train_batch(random.sample(D, BATCH))
        s_t = s_t1
        t += 1
        print(f'TIME STEP {t} / EPSILON {epsilon} / ACTION {action_index} / REWARD {r_t} / Q_MAX {np.max(Q_sa)} / Loss {loss}')
