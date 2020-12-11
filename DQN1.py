from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from collections import deque
import numpy as np
import random
import tensorflow as tf


class DQN:
    MIN_REPLAY_MEMORY_SIZE = 1_000
    MINIBATCH_SIZE = 64
    DISCOUNT = 0.95
    REPLAY_MEMORY_SIZE = 50_000
    checkpoint_filepath = '../results/checkpoint/'

    def __init__(self, shape, output_actions, load):
        self.model = self.create_model(shape, output_actions)
        self.target_model = self.create_model(shape, output_actions)
        if load:
            self.model.load_weights(self.checkpoint_filepath)
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='acc',
            mode='auto')

    def create_model(self, shape, output_actions):
        model = Sequential()
        model.add(Conv2D(256, (2, 2), activation='relu', input_shape=shape))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.2))
        model.add(Conv2D(256, 2, activation='relu'))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(output_actions, activation='relu'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model

    def get_qs(self, state):
        state = np.asarray(state)
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)
        current_qs_list = self.model.predict(current_states)
        X, y = [], []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X) / 255, np.array(y), batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.model_checkpoint_callback] if terminal_state else None)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.MINIBATCH_SIZE * 3:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
