from game import Game
import torch
import numpy as np
from model import Linear_QNet, Qtrainer, ConvNet
from collections import deque
import random

MAX_MEMORY = 10_000
BATCHSIZE = 100
LR = 0.01
EPS = 0.01

device = ("cuda")

class Agent():
    def __init__(self, download_model = False):
        self.game = Game()
        self.model = ConvNet().to(device)#Linear_QNet(inputsize = 400, outputsize = 4)
        if download_model:
            self.model.load_state_dict(torch.load("./surround/model/convmodel.pth"))
        self.trainer = Qtrainer(self.model, LR)
        self.memory = deque(maxlen = MAX_MEMORY)
        self.eps = EPS

    def take_action(self):
        coinA = np.random.binomial(n=1, p = self.eps, size = 1)
        coinB = np.random.binomial(n=1, p = self.eps, size = 1)
        actionA = np.array([0, 0, 0, 0])
        actionB = np.array([0, 0, 0, 0])

        #eps greedy policy for two snakes
        if coinA:
            num = np.random.randint(0, 4, size=1)
            actionA[num] = 1
        else:
            state = self.game.get_state_A()
            state = torch.unsqueeze(state, 0).to(device)
            pred = self.model(state)
            idx = torch.argmax(pred).item()
            actionA[idx] = 1

        if coinB:
            num = np.random.randint(0, 4, size=1)
            actionB[num] = 1
        else:
            state = self.game.get_state_B().to(device)
            pred = self.model(state)
            idx = torch.argmax(pred).item()
            actionB[idx] = 1
        
        return actionA, actionB

    def remember(self, action, state, reward, newstate, end):
        self.memory.append((action, state, reward, newstate, end))

    def train_short_memory(self, action, state, reward, newstate, end):
        self.trainer.train_step(action, state, reward, newstate, end)

    def train_long_memory(self):
        if len(self.memory) > BATCHSIZE:
            #mini_sample_idx = np.random.randint(0, len(self.memory), size = BATCHSIZE)
            mini_sample = random.sample(self.memory, BATCHSIZE)
        else:
            mini_sample = self.memory
        
        actions, states, rewards, newstates, ends = zip(*mini_sample)
        self.trainer.train_step(actions, states, rewards, newstates, ends, long = True)

def train():
    game = Game()
    agent = Agent(download_model=True)
    for _ in range(4000):
        stateA = game.get_state_A()
        stateB = game.get_state_B()

        actionA, actionB = agent.take_action()

        rewardA, rewardB, end = game.take_step(actionA, actionB)

        newstateA = game.get_state_A()
        newstateB = game.get_state_B()

        actionA = torch.tensor(actionA, dtype=torch.float).to(device)
        actionB = torch.tensor(actionB, dtype=torch.float).to(device)

        agent.train_short_memory(actionA, stateA, rewardA, newstateA, end)
        agent.train_short_memory(actionB, stateB, rewardB, newstateB, end)

        

        agent.remember(actionA, stateA, rewardA, newstateA, end)
        agent.remember(actionB, stateB, rewardB, newstateB, end)
        

        if end:
            if game.steps > game.record:
                game.record = game.steps
                print(game.record)
            game.reset()
            agent.train_long_memory()
            game.n_games += 1
        if _ % 1000 == 0:
            print(_ // 1000, '/100')
    agent.model.save()
            

if __name__ == '__main__':
    train()