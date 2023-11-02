import torch

class FlockEnsemble(object):
    def __init__(self, speed, neighborhood_size, separation_size, cohesion_f):
        self.speed = speed
        
        self.neighb_size = neighborhood_size
        self.sep_size = separation_size
        
        if cohesion_f < 0 or cohesion_f > 1:
            raise ValueError(f"cohesion factor must be in [0,1] but got {cohesion_f}")
        self.cohesion_f = cohesion_f
        self.alignment_f = 1-cohesion_f


    def calculate_forces(positions,forces):
        sepforce = self.do_separate_v(see_mask,positions)
        cohforce = self.do_cohere_v(see_mask)
        aliforce = self.do_align_v(see_mask, forces)

        return self.momentum_f * forces + self.separation_f * sepforce + self.cohesion_f*cohforce + self.alignment_f * aliforce
        

