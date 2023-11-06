class FlockEnsemble(object):

    def __init__(self, speed, neighborhood_radius, separation_radius, cohesion_f, use_vision=True, device="cpu"):
        self.speed = torch.tensor(speed)

        self.neighb_radius = neighborhood_radius
        self.sep_radius = separation_radius

        if cohesion_f < 0 or cohesion_f > 1:
            raise ValueError(
                f"cohesion factor must be in [0,1] but got {cohesion_f}")
        self.cohesion_f = cohesion_f
        self.alignment_f = 1 - cohesion_f
        self.use_vision = use_vision
        self.device = device

    def _average_force(self, force, affect_count):
        force_avg = force.sum(axis=1) / affect_count
        # If there are no neighbors, we will divide by 0 (resulting in nan).
        # Replace those with 0.0 so that we have no cohere force.
        return torch.nan_to_num(force_avg, nan=0.0)

    def _deltas(self, positions):
        deltas = positions.unsqueeze(0) - positions.unsqueeze(1)
        dists = torch.linalg.norm(deltas, axis=-1) + 1e-6

        return deltas, dists

    def _see_mask(self, velocities, weights, deltas, dists, radius):
        isNeighb = dists < radius

        # Without vision, large flocks thats collide tend to merge
        # Whereas with vision, large flocks have some collective momentum
        cos_sim = torch.nn.functional.cosine_similarity(
            deltas, velocities.unsqueeze(1), dim=-1)
        # 180 degree vision
        canSee = (cos_sim > 0)

        # Boids should not affect our themselves, so mask out diagonals.
        self_attn = 1 - torch.eye(len(deltas),device=self.device)

        # Forces are summed up by neighborhood, so to weight
        # those forces we use a weighted neighborhood mask.
        weight_matrix = weights.unsqueeze(0)*weights.unsqueeze(1)
        weighted_isNeighb = isNeighb * weight_matrix
        return (weighted_isNeighb * self_attn).unsqueeze(-1), (canSee *
                                                      self_attn).unsqueeze(-1)

    def _clamp_norm(self, force):
        norms = torch.linalg.norm(force, dim=-1, keepdim=True)
        clamped_norm = torch.clamp(norms, min=0, max=1)
        f_norm = torch.nan_to_num(force / norms, nan=0)
        return f_norm * clamped_norm

    def _sum_neighborhood_effect(self,
                                see_mask,
                                effect,
                                use_vision):
        isNeighb, canSee = see_mask

        canEffect = isNeighb
        if use_vision:
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
        inverse_negative = torch.abs(dists - self.sep_radius)
        negative_deltas = normdeltas * -1 * torch.square(inverse_negative)
        return self._sum_neighborhood_effect(see_mask,
                                            negative_deltas,
                                            False)

    def calculate_acceleration_norm(self, positions, velocities, weights):
        deltas, dists = self._deltas(positions)

        sep_mask = self._see_mask(velocities, weights, deltas, dists, self.sep_radius)
        sepforce = self._do_separate(sep_mask, deltas, dists)

        coh_mask = self._see_mask(velocities, weights, deltas, dists, self.neighb_radius)
        cohforce = self._sum_neighborhood_effect(coh_mask, deltas, self.use_vision)
        aliforce = self._sum_neighborhood_effect(coh_mask, velocities, self.use_vision)

        # For moments where the velocities cancel out, it is useful to use
        # clamp norm instead of normalize (so that the norm can be <1)
        accel_norm = torch.nn.functional.normalize(1 * sepforce + self.cohesion_f * cohforce +
                                     self.alignment_f * aliforce)

        return accel_norm

    def do_physics_step(self,positions,velocities,weights=None,dt=.01):
        if not torch.is_tensor(weights):
            weights = torch.ones(len(positions),device=self.device)

        # TODO: the velocities should be normalized before doing any math on them
        weights = weights / torch.max(weights)
        accel_norm = self.calculate_acceleration_norm(positions,velocities,weights)


        # Boids travel at a constant velocity, so velocity is normalized.
        velocities = torch.nn.functional.normalize(velocities + accel_norm * dt * torch.sqrt(self.speed))
        positions = positions + velocities * dt * self.speed

        return positions, velocities