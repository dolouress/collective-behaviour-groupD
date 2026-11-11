#MATLAB herding_model.m
from dataclasses import dataclass
import numpy as np

@dataclass(frozen = True)
class Parameters:
    no_shp: int = 14
    box_length: float = 250.0
    rad_rep_s: float = 2.0
    rad_rep_dog: float = 12.0
    K_atr: int = 10
    k_atr: int = 4
    k_alg: int = 1
    vs: float = 1.0
    v_dog: float = 1.5
    h: float = 0.5
    rho_a: float = 2.0
    rho_d: float = 1.0
    e: float = 0.3
    c: float = 1.5
    alg_str: float = 1.3
    f_n: float | None = None
    pd: float | None = None
    pc: float | None = None
    n_iter: int = 370

    def __post_init__(self):
        if self.f_n is None:
            object.__setattr__(self, 'f_n', self.rad_rep_s * self.no_shp ** (2/3))
        if self.pd is None:
            object.__setattr__(self, 'pd', self.rad_rep_s * np.sqrt(self.no_shp))
        if self.pc is None:
            object.__setattr__(self, 'pc', self.rad_rep_s)
        if not (0 <= self.k_alg <= self.k_atr <= self.K_atr < self.no_shp):
            raise ValueError('Require 0 <= k_alg <= k_atr <= K_atr < no_shp')
        if self.n_iter < 2:
            raise ValueError('n_iter must be at least 2')

def normalize(vector):
    vector = np.asarray(vector, dtype = float)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        return vector/np.linalg.norm(vector)

def herding_model(p: Parameters, rng: np.random.Generator):
    n, steps = p.no_shp, p.n_iter
    theta_pos = 2*np.pi*rng.random()
    str_side = p.box_length*np.array([np.cos(theta_pos), np.sin(theta_pos)])
    pos_s = str_side - 3*p.rad_rep_s*rng.random((n, 2))
    pos_d = str_side - 3*p.rad_rep_dog*rng.random(2)

    theta_s = 2*np.pi*rng.random(n)
    theta_d = 2*np.pi*rng.random()
    vel_s = np.column_stack((np.cos(theta_s), np.sin(theta_s)))
    vel_d = np.array([np.cos(theta_d), np.sin(theta_d)])

    pos_s_dat = np.full((n, 2, steps), np.nan)
    vel_s_dat = np.full((n, 2, steps), np.nan)
    pos_d_dat = np.full((steps, 2), np.nan)
    vel_d_dat = np.full((steps, 2), np.nan)
    spd_d_dat = np.full(steps, np.nan)
    collect_t = np.full(steps, np.nan)
    drive_t = np.full(steps, np.nan)
    force_slow_t = np.full(steps, np.nan)

    pos_s_dat[:, :, 0] = pos_s
    vel_s_dat[:, :, 0] = vel_s
    pos_d_dat[0, :] = pos_d
    vel_d_dat[0, :] = vel_d
    spd_d_dat[0] = p.v_dog

    pos_s_prev, pos_d_prev = pos_s.copy(), pos_d.copy()
    vel_s_prev, vel_d_prev = vel_s.copy(), vel_d.copy()

    for t in range(1, steps):
        for i in range(n):
            dog_vector = pos_d_prev - pos_s_prev[i]
            dog_distance = np.linalg.norm(dog_vector)
            dog_unit = normalize(dog_vector)

            if dog_distance > p.rad_rep_dog:
                offsets = pos_s_prev - pos_s_prev[i]
                distances = np.linalg.norm(offsets, axis=1)
                neighbours = np.flatnonzero(distances < p.rad_rep_s)
                if len(neighbours) > 1:
                    neighbours = neighbours[neighbours != i]
                    rep = -normalize(np.sum(offsets[neighbours]/distances[neighbours, None], axis = 0))
                    next_vel = normalize(p.h*vel_s_prev[i] + p.rho_a*rep)
                    pos_s[i] = pos_s[i] + p.vs*next_vel
                    vel_s_dat[i, :, t] = next_vel
                else:
                    vel_s_dat[i, :, t] = vel_s_prev[i]
            else:
                offsets = pos_s_prev - pos_s_prev[i]
                distances = np.linalg.norm(offsets, axis = 1)
                neighbours = np.flatnonzero(distances < p.rad_rep_s)
                has_repulsion = len(neighbours) > 1
                if has_repulsion:
                    neighbours = neighbours[neighbours != i]
                    rep = -normalize(np.sum(offsets[neighbours]/distances[neighbours, None], axis = 0))

                dog_repulsion = - dog_unit
                near = np.argsort(distances, kind = 'stable')[1:p.K_atr + 1]
                selected = rng.permutation(near)[:p.k_atr]
                attraction = normalize(np.sum(offsets[selected]/(distances[selected, None] + np.finfo(float).eps), axis = 0))
                
                aligned = rng.permutation(selected)[:p.k_alg]
                if p.k_alg == 1:
                    scalar = np.sum(vel_s_prev[aligned])
                    alignment = normalize(scalar)
                else:
                    alignment = normalize(
                        np.sum(vel_s_prev[aligned], axis=0)
                    )

                theta_error = 2*np.pi*rng.random()
                random_unit = np.array([np.cos(theta_error), np.sin(theta_error)])

                candidate = (p.h*vel_s_prev[i] + p.rho_d*dog_repulsion + p.c*attraction + p.e*random_unit + p.alg_str*alignment)
                if has_repulsion:
                    candidate += p.rho_a*rep
                next_vel = normalize(candidate)
                pos_s[i] += p.vs*next_vel
                vel_s_dat[i, :, t] = next_vel

        sheep_from_dog = pos_s_prev - pos_d_prev
        distances_dog = np.linalg.norm(sheep_from_dog, axis = 1)
        if np.min(distances_dog) <= p.rad_rep_s:
            pos_d = pos_d + 0.05*vel_d_prev
            vel_d_dat[t] = vel_d_prev
            spd_d_dat[t] = 0.05
            force_slow_t[t] = 1
        else:
            centre = np.mean(pos_s_prev, axis=0)
            centre_offsets = pos_s_prev - centre
            centre_distances = np.linalg.norm(centre_offsets, axis = 1)
            if np.max(centre_distances) >= p.f_n:
                farthest = np.argmax(centre_distances)
                behind = centre_distances[farthest] + p.pc
                target = centre + behind * normalize(centre_offsets[farthest])
                desired = normalize(target - pos_d_prev)
                theta_error = 2 * np.pi * rng.random()
                error = np.array([np.cos(theta_error), np.sin(theta_error)])
                next_vel = normalize(desired + p.e * error)
                pos_d = pos_d + p.v_dog * next_vel
                collect_t[t] = 1
            else:
                behind = np.linalg.norm(centre) + p.pd
                target = behind * normalize(centre)
                desired = normalize(target - pos_d_prev)
                theta_error = 2*np.pi*rng.random()
                error = np.array([np.cos(theta_error), np.sin(theta_error)])
                next_vel = normalize(desired + p.e * error)
                pos_d = pos_d + p.v_dog * next_vel
                drive_t[t] = 1
            vel_d_prev = next_vel
            vel_d_dat[t] = next_vel
            spd_d_dat[t] = p.v_dog

        pos_s_dat[:, :, t] = pos_s
        vel_s_prev = vel_s_dat[:, :, t].copy()
        pos_s_prev = pos_s.copy()
        pos_d_dat[t] = pos_d
        pos_d_prev = pos_d.copy()

    return dict(pos_s = pos_s_dat, pos_d = pos_d_dat, vel_s = vel_s_dat, vel_d = vel_d_dat, spd_d = spd_d_dat,
            collect_t = collect_t, drive_t = drive_t, force_slow = force_slow_t)
