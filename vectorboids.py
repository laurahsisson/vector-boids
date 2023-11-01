#!/usr/bin/env python3
from random import randint
import pygame as pg
import torch
import numpy as np
'''
PyNBoids - a Boids simulation - github.com/Nikorasu/PyNBoids
Uses numpy array math instead of math lib, more efficient.
Copyright (c) 2021  Nikolaus Stromberg  nikorasu85@gmail.com
'''
FLLSCRN = True          # True for Fullscreen, or False for Window
WRAP = True            # False avoids edges, True wraps to other side
FISH = False            # True to turn boids into fish
WIDTH = 1200            # Window Width (1200)
HEIGHT = 800            # Window Height (800)
BGCOLOR = (0, 0, 0)     # Background color in RGB
SHOWFPS = True         # show frame rate


DEBUG = False

if DEBUG:
    SPEED = 30            # Movement speed
    FPS = 15                # 30-90
    BOIDZ = 100
    NEIGHBSIZE = 100000
    SEPFORCE = 10000
else:
    SPEED = 30
    FPS = 60
    BOIDZ = 150
    NEIGHBSIZE = 80
    SEPFORCE = 500





class Boid(pg.sprite.Sprite):
    def __init__(self, boidNum, data, drawSurf, isFish=False, cHSV=None):
        super().__init__()
        self.data = data
        self.bnum = boidNum
        self.drawSurf = drawSurf
        self.image = pg.Surface((15, 15)).convert()
        self.image.set_colorkey(0)
        self.color = pg.Color(0)  # preps color so we can use hsva
        self.color.hsva = (randint(0,360), 25, 100) if cHSV is None else cHSV # randint(5,55) #4goldfish
        if isFish:  # (randint(120,300) + 180) % 360  #4noblues
            pg.draw.polygon(self.image, self.color, ((7,0),(12,5),(3,14),(11,14),(2,5),(7,0)), width=3)
            self.image = pg.transform.scale(self.image, (16, 24))
        else : pg.draw.polygon(self.image, self.color, ((7,0), (13,14), (7,11), (1,14), (7,0)))
        self.neighbSize = NEIGHBSIZE
        self.orig_image = pg.transform.rotate(self.image.copy(), -90)
        self.dir = pg.Vector2(1, 0)  # sets up forward direction
        maxW, maxH = self.drawSurf.get_size()
        self.rect = self.image.get_rect(center=(randint(50, maxW - 50), randint(50, maxH - 50)))
        self.ang = randint(0, 360)  # random start angle, & position ^
        self.pos = pg.Vector2(self.rect.center)
        self.data.array[self.bnum,:3] = torch.tensor([self.pos[0], self.pos[1], self.ang])
        self.data.forces[self.bnum] = (2*torch.rand((2,)))-1
        self.data.boidz[self.bnum] = self
    

    def draw_delta(self,delta):
        pg.draw.line(self.drawSurf, self.color, self.pos, pg.Vector2(self.pos[0]+delta[0],self.pos[1]+delta[1]), width=1)

    def draw_to(self,pos):
        pg.draw.line(self.drawSurf, self.color, self.pos, pg.Vector2(pos[0],pos[1]), width=1)

    def normalize_vector(self,vector):
        if torch.linalg.norm(vector) == 0:
            return torch.zeros((2,))

        return vector / torch.linalg.norm(vector)

    def do_separate(self, myPos, see_mask, otherPos):
        canSee, deltas = see_mask
        dists = torch.linalg.norm(deltas,axis=1).unsqueeze(-1) + 1e-6
        normdeltas = deltas / dists


        expdeltas = normdeltas / (dists**2)

        sepforce = -1*expdeltas.sum(axis=0)
        return sepforce

    def do_cohere(self, myPos, see_mask, otherPos):
        canSee, deltas = see_mask

        if canSee.sum() == 0:
            return torch.zeros((2,))

        neighbPos = otherPos * canSee  

        if canSee.sum() == 0:
            return torch.zeros((2,))

        neighbCenter = neighbPos.sum(axis=0) / canSee.sum()

        toNeighbCenter =  neighbCenter - myPos

        return self.normalize_vector(toNeighbCenter)

    def do_align(self, myPos, see_mask, otherPos, otherForce):
        canSee, deltas = see_mask

        if canSee.sum() == 0:
            return torch.zeros((2,))

        neighbForces = otherForce * canSee
        alignForce = neighbForces.sum(axis=0) / canSee.sum()

        return self.normalize_vector(alignForce)

    def see_mask_v(self,positions,forces):
        # Need to do like a diagonal 0.
        deltas = positions.unsqueeze(0) - positions.unsqueeze(1)
        dists = torch.linalg.norm(deltas,axis=-1) + 1e-6

        # deltas i, j is delta from i to j (j.pos - i.pos)
        assert torch.all(deltas[0,1] == positions[1] - positions[0])
        
        isNeighb = dists < self.neighbSize
        cos_sim = torch.nn.functional.cosine_similarity(deltas,forces,dim=-1)
        canSee = (cos_sim > 0)

        return (canSee, isNeighb, deltas, dists)


    def do_separate_v(self, see_mask, positions):
        canSee, isNeighb, deltas, dists = see_mask

        normdeltas = deltas / dists.unsqueeze(-1)

        # Diagonals will all be 0, otherwise, the norm of the vectors will be 1.
        norms = torch.linalg.norm(normdeltas,axis=-1)
        assert torch.all(torch.logical_or(torch.isclose(norms,torch.tensor(1.0)), torch.isclose(norms,torch.tensor(0.0))))

        # Separate is not affected by line of sight.
        normdeltas = normdeltas * isNeighb.unsqueeze(-1)      

        expdeltas = normdeltas / torch.square(dists.unsqueeze(-1))

        sepforce = -1*expdeltas.sum(axis=1)
        return sepforce

    def update(self, dt, speed, ejWrap=False):
        if self.bnum != 0:
            return

        positions, forces = self.data.array[:,:2], self.data.forces
        see_mask = self.see_mask_v(positions,forces)
        
        sepforce = self.do_separate_v(see_mask,positions)

        allforce = SEPFORCE*sepforce

        positions += allforce * dt * speed


        # Update data
        for i, b in enumerate(self.data.boidz):
            b.pos = pg.Vector2(positions[i,0],positions[i,1])
            b.draw_delta(allforce[i]*100)
            b.rect.center = b.pos

        self.data.array[:,:2] = positions







class BoidArray():  # Holds array to store positions and angles
    def __init__(self):
        self.array = torch.zeros((BOIDZ, 3))
        self.forces = torch.zeros((BOIDZ, 2))
        self.boidz = [None]*BOIDZ

def main():
    pg.init()  # prepare window
    pg.display.set_caption("PyNBoids")
    try: pg.display.set_icon(pg.image.load("nboids.png"))
    except: print("FYI: nboids.png icon not found, skipping..")
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
        nBoids.add(Boid(n, dataArray, screen, FISH))  # spawns desired # of boidz

    clock = pg.time.Clock()
    if SHOWFPS : font = pg.font.Font(None, 30)

    # main loop
    while True:
        for e in pg.event.get():
            if e.type == pg.QUIT or e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE:
                return

        dt = clock.tick(FPS) / 1000
        screen.fill(BGCOLOR)
        nBoids.update(dt, SPEED, WRAP)
        nBoids.draw(screen)

        if SHOWFPS : screen.blit(font.render(str(int(clock.get_fps())), True, [0,200,0]), (8, 8))

        with torch.no_grad():
            pg.display.update()

if __name__ == '__main__':
    main()  # by Nik
    pg.quit()
