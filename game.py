import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from blob import Blob
from matplotlib import style
import random

style.use("ggplot")



class Game:
    SIZE = 10
    MAX_STEP = 200
    CURRENT_STEP = 0
    ENEMY_PENALTY = 500
    FOOD_REWARD = 200
    MOVE_PENALTY = 10
    RETURN_IMAGES = True

    def __init__(self):
        self.player_color = (255, 0, 0)
        self.enemy_color = (0, 0, 255)
        self.food_color = (0, 255, 0)
        self.board = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        self.player = Blob(self.SIZE)
        self.board[self.player.y, self.player.x] = self.player_color
        self.enemy = self.get_enemy_blob()
        self.board[self.enemy.y, self.enemy.x] = self.enemy_color
        self.food = self.get_food_blob()
        self.board[self.food.y, self.food.x] = self.food_color
        self.target_update_counter = 0

    def get_enemy_blob(self):
        new_blob = Blob(self.SIZE)
        while new_blob.x == self.player.x and new_blob.y == self.player.y:
            new_blob = Blob(self.SIZE)
        return new_blob

    def get_food_blob(self):
        new_blob = Blob(self.SIZE)
        while new_blob.x == self.player.x and new_blob.y == self.player.y and new_blob.x == self.enemy.x and new_blob.y == self.enemy.y:
            new_blob = Blob(self.SIZE)
        return new_blob

    def get_image(self):
        self.board = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        self.board[self.enemy.y, self.enemy.x] = self.enemy_color
        self.board[self.food.y, self.food.x] = self.food_color
        self.board[self.player.y, self.player.x] = self.player_color
        if self.RETURN_IMAGES:
            img = Image.fromarray(self.board, 'RGB')
            return img
        else:
            observation = (self.player - self.food) + (self.player - self.enemy)
            return observation

    def render(self, key):
        img = self.get_image()
        img = img.resize((300, 300))
        cv2.imshow("game", np.array(img))
        cv2.waitKey(key)

    def step(self, action=None):
        self.CURRENT_STEP += 1
        self.player.action(action)

        #### MAYBE ###
        # enemy.move()
        # food.move()
        ##############

        new_observation = np.array(self.get_image())

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.CURRENT_STEP >= self.MAX_STEP:
            done = True

        return new_observation, reward, done
