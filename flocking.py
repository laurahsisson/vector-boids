import torch

class FlockEnsemble(object):

    def __init__(self, speed, neighborhood_size, separation_size, cohesion_f):
        self.speed = torch.tensor(speed)

        self.neighb_size = neighborhood_size
        self.sep_size = separation_size

        if cohesion_f < 0 or cohesion_f > 1:
            raise ValueError(
                f"cohesion factor must be in [0,1] but got {cohesion_f}")
        self.cohesion_f = cohesion_f
        self.alignment_f = 1 - cohesion_f


    def _average_force(self, force, affect_count):
        force_avg = force.sum(axis=1) / affect_count
        # If there are no neighbors, we will divide by 0 (resulting in nan).
        # Replace those with 0.0 so that we have no cohere force.
        return torch.nan_to_num(force_avg, nan=0.0)

    def _deltas(self, positions):
        deltas = positions.unsqueeze(0) - positions.unsqueeze(1)
        dists = torch.linalg.norm(deltas, axis=-1) + 1e-6

        return deltas, dists

    def _see_mask(self, velocities, deltas, dists, size):
        isNeighb = dists < size

        # Without vision, large flocks thats collide tend to merge
        # Whereas with vision, large flocks have some collective momentum
        cos_sim = torch.nn.functional.cosine_similarity(
            deltas, velocities.unsqueeze(1), dim=-1)
        # 180 degree vision
        canSee = (cos_sim > 0)

        # Boids should not affect our themselves, so mask out diagonals.
        self_attn = 1 - torch.eye(len(deltas))

        return (isNeighb * self_attn).unsqueeze(-1), (canSee *
                                                      self_attn).unsqueeze(-1)

    def _clamp_norm(self, force):
        norms = torch.linalg.norm(force, dim=-1, keepdim=True)
        clamped_norm = torch.clamp(norms, min=0, max=1)
        f_norm = torch.nan_to_num(force / norms, nan=0)
        return f_norm * clamped_norm

    def _sum_neighborhood_effect(self,
                                see_mask,
                                effect,
                                use_vison):
        isNeighb, canSee = see_mask

        canEffect = isNeighb
        if use_vison:
            canEffect = canEffect * canSee

        effect_count = canEffect.sum(axis=0)
        neighb_effect = effect * canEffect

        neighb_effect_sum = self._average_force(neighb_effect, effect_count)

        return self._clamp_norm(neighb_effect_sum)

    def _do_separate(self, see_mask, deltas, dists):
        # normdeltas represents a normalized vector from the boid's position towards all neighbors
        dists = dists.unsqueeze(-1)
        normdeltas = deltas / dists

        # subtracting that value from dists results in a repulsion
        # that is quadratically stronger for nearer neighbors.
        inverse_negative = torch.abs(dists - self.sep_size)
        negative_deltas = normdeltas * -1 * torch.square(inverse_negative)
        return self._sum_neighborhood_effect(see_mask,
                                            negative_deltas,
                                            False)

    def calculate_acceleration_norm(self,positions, velocities):
        deltas, dists = self._deltas(positions)

        sep_mask = self._see_mask(velocities, deltas, dists, self.sep_size)
        sepforce = self._do_separate(sep_mask, deltas, dists)

        coh_mask = self._see_mask(velocities, deltas, dists, self.neighb_size)
        cohforce = self._sum_neighborhood_effect(coh_mask, deltas, True)
        aliforce = self._sum_neighborhood_effect(coh_mask, velocities, True)

        # For moments where the velocities cancel out, it is useful to use
        # clamp norm instead of normalize (so that the norm can be <1)
        accel_norm = self._clamp_norm(1 * sepforce + self.cohesion_f * cohforce +
                                     self.alignment_f * aliforce)

        return accel_norm

    def do_physics_step(self,positions,velocities,dt):
        accel_norm = self.calculate_acceleration_norm(positions,velocities)

        # Boids travel at a constant velocity, so velocity is normalized.
        velocities = torch.nn.functional.normalize(velocities + accel_norm * dt * torch.sqrt(self.speed))
        positions = positions + velocities * dt * self.speed

        return positions, velocities

