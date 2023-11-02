import torch

class FlockEnsemble(object):
    def __init__(self, speed, neighborhood_size, momentum_f, separation_f, cohesion_f, alignment_f):
        self.speed = speed
        self.neighb_size = neighborhood_size
        self.momentum_f = momentum_f
        self.separation_f = separation_f
        self.cohesion_f = cohesion_f
        self.alignment_f = alignment_f

    def _average_neighborhood(self,force,is_neighb):
        force_avg = (force * is_neighb).sum(axis=1) / is_neighb.sum(axis=0)
        # If there are no neighbors, we will divide by 0 (resulting in nan).
        # Replace those with 0.0 so that we have no cohere force.
        return torch.nan_to_num(force_avg,nan=0.0)

    def _calculate_see_mask(positions,forces):
        # deltas i, j is delta from i to j (j.pos - i.pos)
        deltas = positions.unsqueeze(0) - positions.unsqueeze(1)
        dists = torch.linalg.norm(deltas,axis=-1) + 1e-6

        is_neighb = dists < self.neighb_size

        # We should not be our own neighbor, so set all
        # elements along the diagonal to 0.
        self_attn = 1 - torch.eye(len(deltas))
        is_neighb = is_neighb * self_attn

        return (deltas, is_neighb.unsqueeze(-1), dists.unsqueeze(-1))

    def _do_separate(self, see_mask, positions):
        deltas, is_neighb, dists = see_mask

        normdeltas = deltas / dists
        normdeltas = normdeltas * is_neighb      

        expdeltas = normdeltas / torch.square(dists)

        return -1*expdeltas.sum(axis=1)

     def _do_cohere(self, see_mask):
        deltas, is_neighb, dists = see_mask
        
        to_neighb_center = self._average_neighborhood(neighb_deltas,is_neighb)
        return torch.nn.functional.normalize(to_neighb_center,dim=-1)

    def _do_align(self, see_mask, forces):
        deltas, is_neighb, dists = see_mask

        total_forces = self._average_neighborhood(forces,is_neighb)
        return torch.nn.functional.normalize(total_forces,dim=-1)

    def calculate_forces(positions,forces):
        sepforce = self.do_separate_v(see_mask,positions)
        cohforce = self.do_cohere_v(see_mask)
        aliforce = self.do_align_v(see_mask, forces)

        return self.momentum_f * forces + self.separation_f * sepforce + self.cohesion_f*cohforce + self.alignment_f * aliforce
        

