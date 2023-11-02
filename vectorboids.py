#!/usr/bin/env python3
from random import randint
import pygame as pg
import torch
import numpy as np
import math
import flocking
'''
VectorBoids - Multidimensional Boid Simulation in Pytorch
Simulation by Laura Sisson
Pygame Framework and UI by Nikolaus Stromberg
'''

# UI SETTINGS
FLLSCRN = True  # True for Fullscreen, or False for Window
WRAP = True  # False avoids edges, True wraps to other side
FISH = False  # True to turn boids into fish
WIDTH = 1200  # Window Width (1200)
HEIGHT = 800  # Window Height (800)
BGCOLOR = (0, 0, 0)  # Background color in RGB
FPS = 60
SHOWFPS = True  # show frame rate
SATURATION = 25 # For hsv of boids

# SIMULATION SETTINGS
SPEED = 150 # How quickly the boids move and accelerate
BOIDZ = 600 # How many agents
NEIGHBSIZE = 80 # Boids try to cohere up to this distance
SEPSIZE = 30 # Boids try to keep this amount of separation

# Weight balancing cohesion and alignment. The higher this value,
# the more flocking is based on position as opposed to heading. 
COHESION_F = .8
assert 0 <= COHESION_F and COHESION_F <= 1

DIMENSION = 2 # How many dimensions to simulate
assert DIMENSION >= 2 # Does not support 1 dimensional simulations.

# x and y are wrapped in the screen space.
# higher dimensions are wrapped in a space from [-WRAP_EXTRA_DIM,WRAP_EXTRA_DIM]
WRAP_EXTRA_DIM = 100 

class Boid(pg.sprite.Sprite):

    def __init__(self, boidNum, data, drawSurf, cHSV=None):
        super().__init__()
        self.data = data
        self.bnum = boidNum
        self.drawSurf = drawSurf

        self.image = pg.Surface((15, 15)).convert()
        self.image.set_colorkey(0)
        self.color = pg.Color(0)  # preps color so we can use hsva
        self.color.hsva = (
            randint(0, 360), SATURATION,
            100) if cHSV is None else cHSV  # randint(5,55) #4goldfish

        if FISH:  # (randint(120,300) + 180) % 360  #4noblues
            pg.draw.polygon(self.image,
                            self.color, ((7, 0), (12, 5), (3, 14), (11, 14),
                                         (2, 5), (7, 0)),
                            width=3)
            self.image = pg.transform.scale(self.image, (16, 24))
        else:
            pg.draw.polygon(self.image, self.color,
                            ((7, 0), (13, 14), (7, 11), (1, 14), (7, 0)))

        self.orig_image = pg.transform.rotate(self.image.copy(), -90)
        maxW, maxH = self.drawSurf.get_size()
        self.rect = self.image.get_rect(center=(randint(50, maxW - 50),
                                                randint(50, maxH - 50)))
        two_dim_pos = torch.tensor([self.rect.center[0], self.rect.center[1]])
        higher_dim_pos = torch.rand(DIMENSION - 2)

        self.data.positions[self.bnum] = torch.cat([two_dim_pos,higher_dim_pos])
        self.data.velocities[self.bnum] = (2 * torch.rand((DIMENSION, ))) - 1
        self.data.boidz[self.bnum] = self

    def draw_to(self, pos):
        pg.draw.line(self.drawSurf,
                     self.color,
                     self.pos,
                     pg.Vector2(pos[0], pos[1]),
                     width=1)

    def draw_delta(self, delta):
        self.draw_to(pg.Vector2(self.pos[0] + delta[0],
                                self.pos[1] + delta[1]))

    def update(self, dt):
        # Only a single boid handles all calculations.
        # Could def refactor this so that BoidArray does all this math.
        if self.bnum != 0:
            return

        positions, velocities = self.data.flock_ensemble.do_physics_step(self.data.positions,self.data.velocities,dt)

        angles = torch.rad2deg(torch.atan2(velocities[:, 1], velocities[:, 0]))

        maxW, maxH = self.drawSurf.get_size()
        # Wrap x
        positions[:, 0] = positions[:, 0] % maxW
        # Wrap y
        positions[:, 1] = positions[:, 1] % maxH
        # Wrap higher dimensions
        positions[:, 2:] = positions[:, 2:] % WRAP_EXTRA_DIM

        self.data.positions = positions
        self.data.velocities = velocities

        # Update data
        for i, b in enumerate(self.data.boidz):
            b.rect.center = pg.Vector2(positions[i,0],positions[i,1])
            b.image = pg.transform.rotate(b.orig_image, -angles[i])


class BoidArray():  # Holds positions to store positions and angles

    def __init__(self):
        self.positions = torch.zeros((BOIDZ, DIMENSION))
        self.velocities = torch.zeros((BOIDZ, DIMENSION))
        self.boidz = [None] * BOIDZ
        self.flock_ensemble = flocking.FlockEnsemble(SPEED,NEIGHBSIZE,SEPSIZE,COHESION_F)


def main():
    pg.init()  # prepare window
    pg.display.set_caption("PyNBoids")
    try:
        pg.display.set_icon(pg.image.load("nboids.png"))
    except:
        print("FYI: nboids.png icon not found, skipping..")
    # setup fullscreen or window mode
    if FLLSCRN:
        currentRez = (pg.display.Info().current_w, pg.display.Info().current_h)
        screen = pg.display.set_mode(currentRez, pg.SCALED)
        pg.mouse.set_visible(False)
    else:
        screen = pg.display.set_mode((WIDTH, HEIGHT), pg.RESIZABLE)

    nBoids = pg.sprite.Group()
    dataArray = BoidArray()
    for n in range(BOIDZ):
        nBoids.add(Boid(n, dataArray, screen))  # spawns desired # of boidz

    clock = pg.time.Clock()
    if SHOWFPS: font = pg.font.Font(None, 30)

    # main loop
    while True:
        for e in pg.event.get():
            if e.type == pg.QUIT or e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE:
                return

        dt = clock.tick(FPS) / 1000
        screen.fill(BGCOLOR)
        nBoids.update(dt)
        nBoids.draw(screen)

        if SHOWFPS:
            screen.blit(
                font.render(str(int(clock.get_fps())), True, [0, 200, 0]),
                (8, 8))

        with torch.no_grad():
            pg.display.update()


if __name__ == '__main__':
    main()  # by Nik
    pg.quit()
