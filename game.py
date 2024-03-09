import pygame
import torch
import numpy as np
from copy import deepcopy

device = ("cuda")
class Snake():
    def __init__(self, head):
        self.head = head
    
    def set_head(self, head):
        self.head[0] = head[0]
        self.head[1] = head[1]
    
    def move_head(self, action):
        if action[0] == 1:
            self.head += np.array([-1, 0])
        elif action[1] == 1:
            self.head += np.array([0, 1])
        elif action[2] == 1:
            self.head += np.array([1, 0])
        else:
            self.head += np.array([0, -1])


class Game():
    def __init__(self, humanplay = False):
        self.n_games = 0
        self.record = 0
        self.height = 20
        self.steps = 0
        self.width = 20
        self.field_agent = np.zeros([self.height, self.width])
        headA = np.array([self.height // 2 + 1, self.width // 3])
        headB = np.array([self.height // 2 + 1, 2 * self.width // 3])
        self.snakeA = Snake(headA)
        self.snakeB = Snake(headB)
        self.humanplay = humanplay

        self.field_agent[headA[0], headA[1]] = 1
        self.field_agent[headB[0], headB[1]] = 1
        self.field_agent[:, 0] = 1
        self.field_agent[:, -1] = 1
        self.field_agent[0, :] = 1
        self.field_agent[-1, :] = 1

        if humanplay:
            self.field = np.zeros([self.height, self.width])
            self.field[headA[0], headA[1]] = 1
            self.field[headB[0], headB[1]] = 2
            self.field[:, 0] = 3
            self.field[:, -1] = 3
            self.field[0, :] = 3
            self.field[-1, :] = 3


        #for i in range(self.height):
        #    print(*self.field_agent[i])

    def reset(self):
        headA = np.array([self.height // 3, self.width // 2])
        headB = np.array([2 * self.height // 3, self.width // 2])
        self.snakeA.set_head(headA)
        self.snakeB.set_head(headB)
        self.steps = 0
        self.field_agent[1:-1, 1:-1] = 0
        self.field_agent[headA[0], headA[1]] = 1
        self.field_agent[headB[0], headB[1]] = 1

    def get_state_A(self):
        """
        state = torch.from_numpy(np.array(np.concatenate((self.field_agent.reshape(-1),
                                                        self.snakeA.head.reshape(-1), 
                                                        self.snakeB.head.reshape(-1))), dtype=np.float32)).to(device)
        """
        headA = self.snakeA.head
        headB = self.snakeB.head
        state_field = deepcopy(self.field_agent)
        state_field = state_field.astype(np.float32)
        state_field[headA[0], headA[1]] = 10
        state_field[headB[0], headB[1]] = -10
        state = torch.from_numpy(state_field[np.newaxis, :, :])
        return state
    
    def get_state_B(self):
        """
        state = torch.from_numpy(np.array(np.concatenate((self.field_agent.reshape(-1),
                                                        self.snakeB.head.reshape(-1), 
                                                        self.snakeA.head.reshape(-1))), dtype=np.float32)).to(device)
        """

        headA = self.snakeA.head
        headB = self.snakeB.head
        state_field = deepcopy(self.field_agent)
        state_field = state_field.astype(np.float32)
        state_field[headA[0], headA[1]] = -10
        state_field[headB[0], headB[1]] = 10
        state = torch.from_numpy(state_field[np.newaxis, :, :])                                    
        return state

    def take_step(self, actionA, actionB):
        self.steps += 1
        rewardA = 0
        rewardB = 0
        
        self.snakeA.move_head(actionA)
        self.snakeB.move_head(actionB)

        headA = self.snakeA.head
        headB = self.snakeB.head

        collisionA = False
        collisionB = False
        collisionAB = False

        if self.field_agent[headA[0], headA[1]] == 1:
            rewardA = -100
            rewardB = 10
            collisionA = True
        if self.field_agent[headB[0], headB[1]] == 1:
            rewardB = -100
            rewardA = 10
            collisionB = True
        if headA[0] == headB[0] and headA[1] == headB[1]:
            rewardA = -50
            rewardB = -50
            collisionAB = True
        
        self.field_agent[headA[0], headA[1]] = 1
        self.field_agent[headB[0], headB[1]] = 1
        if self.humanplay:
            self.field[headA[0], headA[1]] = 1
            self.field[headB[0], headB[1]] = 2

        end = False
        if collisionA or collisionB or collisionAB:
            end = True
        
        return rewardA, rewardB, end
