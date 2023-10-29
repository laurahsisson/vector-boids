#!/usr/bin/env python3
from random import randint
import pygame as pg
import numpy as np
'''
PyNBoids - a Boids simulation - github.com/Nikorasu/PyNBoids
Uses numpy array math instead of math lib, more efficient.
Copyright (c) 2021  Nikolaus Stromberg  nikorasu85@gmail.com
'''
FLLSCRN = True          # True for Fullscreen, or False for Window
BOIDZ = 100            # How many boids to spawn, too many may slow fps
WRAP = True            # False avoids edges, True wraps to other side
FISH = False            # True to turn boids into fish
WIDTH = 1200            # Window Width (1200)
HEIGHT = 800            # Window Height (800)
BGCOLOR = (0, 0, 0)     # Background color in RGB
SHOWFPS = False         # show frame rate


DEBUG = False

if DEBUG:
    SPEED = 1e-10             # Movement speed
    FPS = 1                # 30-90
else:
    SPEED = 150
    FPS = 60



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
        self.neighbSize = 10
        self.orig_image = pg.transform.rotate(self.image.copy(), -90)
        self.dir = pg.Vector2(1, 0)  # sets up forward direction
        maxW, maxH = self.drawSurf.get_size()
        self.rect = self.image.get_rect(center=(randint(50, maxW - 50), randint(50, maxH - 50)))
        self.ang = randint(0, 360)  # random start angle, & position ^
        self.pos = pg.Vector2(self.rect.center)
        self.data.array[self.bnum,:3] = [self.pos[0], self.pos[1], self.ang]
    

    def draw_delta(self,delta):
        pg.draw.line(self.drawSurf, self.color, self.pos, pg.Vector2(self.pos[0]+delta[0],self.pos[1]+delta[1]), width=1)

    def draw_to(self,pos):
        pg.draw.line(self.drawSurf, self.color, self.pos, pg.Vector2(pos[0],pos[1]), width=1)

    def normalize_vector(self,vector):
        if np.linalg.norm(vector) == 0:
            return np.zeros((2,))

        return vector / np.linalg.norm(vector)

    def do_separate(self, myPos, see_mask, otherPos):
        canSee, deltas = see_mask
        dists = np.expand_dims(np.linalg.norm(deltas,axis=1),-1) + 1e-6
        normdeltas = deltas * canSee / dists


        expdeltas = normdeltas / (dists**2)

        sepforce = -1*expdeltas.sum(axis=0)
        return sepforce

    def do_cohere(self, myPos, see_mask, otherPos):
        canSee, deltas = see_mask

        if canSee.sum() == 0:
            return np.zeros((2,))

        neighbPos = otherPos * canSee  

        if canSee.sum() == 0:
            return np.zeros((2,))

        neighbCenter = neighbPos.sum(axis=0) / canSee.sum()

        toNeighbCenter =  neighbCenter - myPos

        return self.normalize_vector(toNeighbCenter)

    def do_align(self, myPos, see_mask, otherPos, otherForce):
        canSee, deltas = see_mask

        if canSee.sum() == 0:
            return np.zeros((2,))

        neighbForces = otherForce * canSee
        alignForce = neighbForces.sum(axis=0) / canSee.sum()

        return self.normalize_vector(alignForce)

    def see_mask(self, myPos, meforce, otherPos):
        deltas = otherPos-myPos
        dists = np.expand_dims(np.linalg.norm(deltas,axis=1),-1) + 1e-6
        
        isNeighb = dists < self.neighbSize

        cos_sim = np.dot(deltas,meforce)/(np.linalg.norm(deltas)*np.linalg.norm(meforce))
        canSee = np.expand_dims(cos_sim > 0,-1)

        seeDeltas = deltas*canSee*isNeighb
        # if self.bnum < 25:
        #     for d in seeDeltas:
        #         self.draw_delta(d*.33)

        return (canSee, deltas)

    def normalize_random(self,vector):
        if not np.linalg.norm(vector):
            return (2*np.random.random_sample((2,)))-1
        else:
            return vector / np.linalg.norm(vector)

    def update(self, dt, speed, ejWrap=False):
        # pg.draw.line(self.drawSurf, self.color, self.pos, pg.Vector2(self.pos[0],self.pos[1]+25), width=1)
        otherPos = np.delete(self.data.array, self.bnum, 0)
        otherPos = np.array(otherPos)[:,:2]

        otherForce = np.delete(self.data.forces, self.bnum, 0)
        otherForce = np.array(otherPos)[:,:2]

        myPos = np.array([self.pos[0],self.pos[1]])

        meforce = self.data.forces[self.bnum]
        meforce = self.normalize_random(meforce)

        if self.bnum < 10:
            self.draw_delta(meforce*100)

        see_mask = self.see_mask(myPos, meforce, otherPos)

        sepforce = self.do_separate(myPos, see_mask, otherPos)
        cohforce = self.do_cohere(myPos, see_mask, otherPos)
        align = self.do_align(myPos, see_mask, otherPos, otherForce)

        allforce = meforce + 500 * sepforce + cohforce + 5 * align

        # allforce = meforce
        allforce = self.normalize_random(allforce)

        myPos += allforce * dt * speed

        # Actually update position of boid
        self.pos = pg.Vector2(myPos[0],myPos[1])
        
        maxW, maxH = self.drawSurf.get_size()
        if ejWrap and not self.drawSurf.get_rect().contains(self.rect):
            if self.rect.bottom < 0 : self.pos.y = maxH - 10
            elif self.rect.top > maxH : self.pos.y = 10
            if self.rect.right < 0 : self.pos.x = maxW - 10
            elif self.rect.left > maxW : self.pos.x = 10

        self.rect.center = self.pos




        # Finally, output pos/ang to array
        self.data.array[self.bnum,:3] = [self.pos[0], self.pos[1], self.ang]
        self.data.forces[self.bnum] = allforce


class BoidArray():  # Holds array to store positions and angles
    def __init__(self):
        self.array = np.zeros((BOIDZ, 4), dtype=float)
        self.forces = np.zeros((BOIDZ, 2), dtype=float)

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

        pg.display.update()

if __name__ == '__main__':
    main()  # by Nik
    pg.quit()
