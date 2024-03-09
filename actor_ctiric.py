import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt



#actor_network

WIN_REWARD = 1
LOSE_REWARD = -1
PENALTY_REWARD = -5

NUM_EPISODES = 3000
MAX_STEPS = 20
gamma = 0.99
EPS = 0.06


class GridWolrd():
    def __init__(self, height, width):
        self.width = width
        self.height = height
        self.field = np.zeros((height + 2, width + 2))
        self.agent_coord = np.array([1, 1])

        #filling boarders
        self.field[0, :] = PENALTY_REWARD
        self.field[:, 0] = PENALTY_REWARD
        self.field[-1, :] = PENALTY_REWARD
        self.field[:, -1] = PENALTY_REWARD 

    def set_win_coord(self, coord):
        self.field[coord[0], coord[1]] = WIN_REWARD
    
    def set_lose_coord(self, coord):
        self.field[coord[0], coord[1]] = LOSE_REWARD
    
    def set_agent_start_point(self, coord):
        self.agent_coord = coord


    def print_field(self, show_agent=True):
        for i in range(1, self.height + 1):
            for j in range(1, self.width + 1):
                if show_agent and i == self.agent_coord[0] and j == self.agent_coord[1]:
                    print(' a', end = ' ')
                else:
                    print('{:2}'.format(int(self.field[i, j])), end = ' ')
            print('')
    
    def return_state(self):
        state = self.field.copy()
        state[self.agent_coord[0], self.agent_coord[1]] = 10
        return state
    
    def process_action(self, action):
        #action should be one of these: [0, 1], [0, -1], [1, 0], [-1, 0]
        self.agent_coord += action
        done = False
        reward = self.field[self.agent_coord[0], self.agent_coord[1]]
        if reward == PENALTY_REWARD or reward == WIN_REWARD:
            done = True

        return reward, done

    def return_state_space(self):
        return self.field.shape[0] * self.field.shape[1] + 2


def reset_game(height, width):
    game = GridWolrd(height, width)
    game.set_win_coord([3, 3]) 
    lose_coord_list = [[2, 2], [3, 2], [4, 2], [5, 4], [4, 4]]
    for elem in lose_coord_list:  
        game.set_lose_coord(elem)

    return game

def letter_to_action(letter):
    action = [0, 0]
    if letter == 'w':
        action[0] = -1
    elif letter == 'a':
        action[1] = -1
    elif letter == 'd':
        action[1] = 1
    elif letter == 's':
        action[0] = 1
    return action

#now let's make a real agent

class PolicyNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU()
        )
        self.l = nn.Sequential(
            nn.Flatten(start_dim = 0),
            nn.Linear(21 * 32, 4)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.l(x)
        action_prob = F.softmax(x, dim = 0)
        #print(action_prob)
        return action_prob


class StateValueNet(nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU()
        )
        self.l = nn.Sequential(
            nn.Flatten(start_dim = 0),
            nn.Linear(21 * 32, 1)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.l(x)
        return x.unsqueeze(dim = 0)


class Agent():
    def __init__(self, state_space, action_space):
        self.actor = PolicyNet(state_space, action_space) 
        self.critic = StateValueNet(state_space)
        self.actor_opt = torch.optim.SGD(self.actor.parameters(), lr = 0.01)
        self.critic_opt = torch.optim.SGD(self.critic.parameters(), lr = 0.01)

    def take_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(dim = 0)

        action_probs = self.actor(state)
        m = Categorical(action_probs)
        raw_action = m.sample()
        #adding extra exploration
        if np.random.binomial(n = 1, p = EPS, size = 1):
            raw_action = torch.from_numpy(np.random.choice([0, 1, 2, 3], size = 1))
        #print(raw_action.item() == 0)
        action = np.array([0, 0])

        if raw_action.item() == 0:
            action[0] = -1
        elif raw_action.item() == 1:
            action[1] = -1
        elif raw_action.item() == 2:
            action[1] = 1
        elif raw_action.item() == 3:
            action[0] = 1
        
        return action, m.log_prob(raw_action)
    
    def return_state_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(dim = 0)
        return self.critic(state)



def main():
    max_score = 0
    scores = []
    height, width = 5, 9
    game = reset_game(height, width)
    agent = Agent(game.return_state_space(), 4)

    #setting goal and obstacles
    #coord start with 1 and 1
    

    done = False
    for episode in range(NUM_EPISODES):
        game = reset_game(height, width)
        done = False
        score = 0
        state = game.return_state()
        I = 1
        for step in range(MAX_STEPS):
            
            #посчитать оба лосса
            action, log_prob = agent.take_action(state)
            reward, done = game.process_action(action)
            new_state = game.return_state()
            score += reward + 0.01

            state_val = agent.return_state_value(state)
            new_state_val = agent.return_state_value(state)

            if done:
                new_state_val = torch.tensor([0]).float().unsqueeze(dim = 0)
            
            critic_loss = F.mse_loss(reward + gamma * new_state_val, state_val)
            critic_loss *= I

            advantage = reward + gamma * new_state_val.item() - state_val.item()
            actor_loss = -log_prob * advantage
            actor_loss *= I

            actor_loss.backward()
            agent.actor_opt.step()
            agent.actor_opt.zero_grad()

            critic_loss.backward()
            agent.critic_opt.step()
            agent.critic_opt.zero_grad()

            if done:
                break
            I *= gamma
            state = new_state
        #score /= (step+1)
        if score > max_score:
            print(score)
            max_score = score
        scores.append(score)

            
    print(*scores)   

if __name__ == '__main__':
    main()