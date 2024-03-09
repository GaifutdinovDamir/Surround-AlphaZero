import pygame
import numpy as np
import torch
from game import Game
from model import Linear_QNet, ConvNet


HEIGHT, WIDTH = 20 * 20, 20 * 20
SNAKE_BOX_SIZE = 20

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 120)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
BROWN = (80, 42, 15)
START_FPS = 60

HIT_A = pygame.USEREVENT + 1
HIT_B = pygame.USEREVENT + 1


pygame.font.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Surround")

def get_human_action(keys_pressed, field, head, prev_human_action):
    action = np.array([0, 0, 0, 0])
    if keys_pressed[pygame.K_w] and field[head[0] - 1, head[1]] == 0:
        action[0] = 1
    elif keys_pressed[pygame.K_d] and field[head[0], head[1]+1] == 0:
        action[1] = 1
    elif keys_pressed[pygame.K_s] and field[head[0]+1, head[1]] == 0:
        action[2] = 1
    elif keys_pressed[pygame.K_a] and field[head[0], head[1]-1] == 0:
        action[3] = 1
    else:
        action[:] = prev_human_action[:]
    return action


def draw_window(field):
    WIN.fill(WHITE)
    #Snake
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            box = pygame.Rect(20 * j, 20 * i, SNAKE_BOX_SIZE, SNAKE_BOX_SIZE)
            if field[i, j] == 1:
                pygame.draw.rect(WIN, GREEN, box)
            elif field[i, j] == 2:
                pygame.draw.rect(WIN, RED, box)
            elif field[i, j] == 3:
                pygame.draw.rect(WIN, BROWN, box)
    pygame.display.update()


def main():
    model = ConvNet()
    model.load_state_dict(torch.load("./surround/model/convmodel.pth"))
    game = Game(humanplay = True)
    clock = pygame.time.Clock()
    run = True
    time = 0
    prev_human_action = np.array([0, 1, 0, 0])
    while run:
        clock.tick(START_FPS)
        time += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
            if event.type == HIT_A or event.type == HIT_B:
                run = False
                break

        keys_pressed = pygame.key.get_pressed()
        human_action = get_human_action(keys_pressed, game.field, game.snakeA.head, prev_human_action)
        if time % 6 == 0:
            state = game.get_state_B()
            #greedy policy
            model_action = np.array([0, 0, 0, 0])
            model_action[torch.argmax(model(state)).item()] = 1

            rw1, rw2, end = game.take_step(human_action, model_action)
            if end:
                pygame.event.post(pygame.event.Event(HIT_A))
            time = 0 
            prev_human_action[:] = human_action[:]
        draw_window(game.field)
   
    WIN.fill(WHITE)
    #draw_text("To start a new game press Q")
    wait = True
    while wait:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        if pygame.key.get_pressed()[pygame.K_q] == 1:
            wait = False
    main()


main()