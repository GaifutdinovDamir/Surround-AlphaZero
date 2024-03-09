import pygame
import numpy as np
import random 

HEIGHT, WIDTH = 680, 1200
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

class snake():
    def __init__(self, head, player):
        self.head = np.array(head)
        self.player = player

        self.speed = 'Up'
        coin = random.randint(0, 3)
        if coin == 1:
            self.speed = 'Down'
        elif coin == 2:
            self.speed = 'Right'
        else: 
            self.speed = 'Left'
    
    def update_speed(self, keys_pressed, field):
        if self.player == 'A':
            if keys_pressed[pygame.K_w] and field[self.head[0], self.head[1]-1] == 0:
                self.speed = 'Up'
            elif keys_pressed[pygame.K_s] and field[self.head[0], self.head[1]+1] == 0:
                self.speed = "Down"
            elif keys_pressed[pygame.K_d] and field[self.head[0]+1, self.head[1]] == 0:
                self.speed = 'Right'
            elif keys_pressed[pygame.K_a] and field[self.head[0]-1, self.head[1]] == 0:
                self.speed = 'Left'
        elif self.player == 'B':
            if keys_pressed[pygame.K_UP] and field[self.head[0], self.head[1]-1] == 0:
                self.speed = 'Up'
            elif keys_pressed[pygame.K_DOWN] and field[self.head[0], self.head[1]+1] == 0:
                self.speed = "Down"
            elif keys_pressed[pygame.K_RIGHT] and field[self.head[0]+1, self.head[1]] == 0:
                self.speed = 'Right'
            elif keys_pressed[pygame.K_LEFT] and field[self.head[0]-1, self.head[1]] == 0:
                self.speed = 'Left'


    def move(self, field):
        new_head = np.array([self.head[0], self.head[1]])
        if self.speed == 'Up':
            new_head[1] = self.head[1] - 1
        elif self.speed == 'Down':
            new_head[1] = self.head[1] + 1
        elif self.speed == 'Right':
            new_head[0] = self.head[0] + 1 
        elif self.speed == 'Left':
            new_head[0] = self.head[0] - 1
        
        if field[new_head[0], new_head[1]] != 0:
            if self.player == 'A':
                pygame.event.post(pygame.event.Event(HIT_A))
            else:
                pygame.event.post(pygame.event.Event(HIT_B))
        self.head[0], self.head[1] = new_head[0], new_head[1]
        if self.player == 'A':
            field[self.head[0], self.head[1]] = 1
        elif self.player == 'B':
            field[self.head[0], self.head[1]] = 2


def draw_window(field):
    WIN.fill(WHITE)
    #Snake
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            box = pygame.Rect(20 * i, 20 * j, SNAKE_BOX_SIZE, SNAKE_BOX_SIZE)
            if field[i, j] == 1:
                pygame.draw.rect(WIN, GREEN, box)
            elif field[i, j] == 2:
                pygame.draw.rect(WIN, RED, box)
            elif field[i, j] == 3:
                pygame.draw.rect(WIN, BROWN, box)
    pygame.display.update()

def main():
    clock = pygame.time.Clock()
    nrows = HEIGHT // SNAKE_BOX_SIZE
    ncols = WIDTH // SNAKE_BOX_SIZE
    field = np.zeros(shape=(ncols, nrows))
    field[0, :] = 3
    field[-1, :] = 3
    field[:, 0] = 3
    field[:, -1] = 3

    snakeA = snake([12, 10], 'A')
    field[12, 10] = 1
    snakeB = snake([30, 10], 'B')
    field[30, 10] = 2
    
    run = True
    time = 0
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
        snakeA.update_speed(keys_pressed, field)
        snakeB.update_speed(keys_pressed, field)
        if time % 6 == 0:
            snakeA.move(field)
            snakeB.move(field)
            time = 0 
        draw_window(field)
   
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


if __name__ == '__main__':
    main()